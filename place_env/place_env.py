import os
import math
import random
import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import time
import hashlib
import colorsys
from itertools import combinations

from util import timeit

from placedb import PlaceDB, LefDefPlaceDB

class PlaceEnv(gym.Env):

    def __init__(self, placedb: PlaceDB, grid = 224, ignore_next = False):
        print(f"{grid=}, {grid * grid=}")
        print(f"macro num: {len(placedb.macro_info)}")
        print(f"cell num: {len(placedb.cell_info)}")
        print(f"node num: {len(placedb.node_info)}")
        print(f"net num: {len(placedb.net_info)}")

        assert grid * grid >= len(placedb.macro_info), "grid size is too small"
        self.grid = grid
        self.ignore_next = ignore_next
        self.canvas_width, self.canvas_height = placedb.canvas_size
        self.placedb = placedb
        self.num_macro = len(placedb.macro_info)
        self.placed_num_macro = len(placedb.macro_place_queue)
        self.num_net = len(placedb.net_info)
        self.node_name_list = placedb.macro_place_queue
        self.action_space = spaces.Discrete(self.grid * self.grid)
        state_dim = 5 * self.grid * self.grid + 3
        self.observation_space = spaces.Box(low = -10, high = 10, shape = (state_dim,), dtype = np.float32)

        self.state = None
        self.net_min_max_ord = {}
        self.node_pos = {}
        self.net_placed_set = None
        self.last_reward = 0
        self.num_macro_placed = None
        self.node_x_max = 0
        self.node_x_min = self.grid
        self.node_y_max = 0
        self.node_y_min = self.grid
        self.ratio = self.canvas_height / self.grid
        print("self.ratio = {:.2f}".format(self.ratio))
    
    def reset(self):
        self.num_macro_placed = 0
        canvas = np.zeros((self.grid, self.grid))
        self.node_pos = {}
        self.net_min_max_ord = {}
        self.net_fea = np.zeros((self.num_net, 4))
        self.net_fea[:, 0] = 0
        self.net_fea[:, 1] = 1.0
        self.net_fea[:, 2] = 0
        self.net_fea[:, 3] = 1.0
        self.rudy = np.zeros((self.grid, self.grid))
        if os.getenv('PLACEENV_IGNORE_PORT', '0') == '1':
            print('`PLACEENV_IGNORE_PORT=1`, so chip ports are ignored')
        else:
            for port_name in self.placedb.port2net_dict:
                for net_name in self.placedb.port2net_dict[port_name]:
                    pin_x = round(self.placedb.port_info[port_name]['coordinate'][0] / self.ratio)
                    pin_y = round(self.placedb.port_info[port_name]['coordinate'][1] / self.ratio)
                    if net_name in self.net_min_max_ord:
                        if pin_x > self.net_min_max_ord[net_name]['max_x']:
                            self.net_min_max_ord[net_name]['max_x'] = pin_x
                            self.net_fea[self.placedb.net_info[net_name]['id']][1] = pin_x / self.grid
                        elif pin_x < self.net_min_max_ord[net_name]['min_x']:
                            self.net_min_max_ord[net_name]['max_y'] = pin_y
                            self.net_fea[self.placedb.net_info[net_name]['id']][0] = pin_x / self.grid
                        if pin_y > self.net_min_max_ord[net_name]['max_y']:
                            self.net_min_max_ord[net_name]['max_y'] = pin_y
                            self.net_fea[self.placedb.net_info[net_name]['id']][3] = pin_y / self.grid
                        elif pin_y < self.net_min_max_ord[net_name]['min_y']:
                            self.net_min_max_ord[net_name]['min_y'] = pin_y
                            self.net_fea[self.placedb.net_info[net_name]['id']][2] = pin_y / self.grid
                    else:
                        self.net_min_max_ord[net_name] = {}
                        self.net_min_max_ord[net_name]['max_x'] = pin_x
                        self.net_min_max_ord[net_name]['min_x'] = pin_x
                        self.net_min_max_ord[net_name]['max_y'] = pin_y
                        self.net_min_max_ord[net_name]['min_y'] = pin_y
                        self.net_fea[self.placedb.net_info[net_name]['id']][1] = pin_x / self.grid
                        self.net_fea[self.placedb.net_info[net_name]['id']][0] = pin_x / self.grid
                        self.net_fea[self.placedb.net_info[net_name]['id']][3] = pin_y / self.grid
                        self.net_fea[self.placedb.net_info[net_name]['id']][2] = pin_y / self.grid

        self.net_placed_set :dict[str, set] = {}
        self.num_macro_placed = 0
        
        net_img = np.zeros((self.grid, self.grid))
        net_img_2 = np.zeros((self.grid, self.grid))

        next_x = math.ceil(max(1, self.placedb.node_info[self.node_name_list[self.num_macro_placed]]['width'] / self.ratio))
        next_y = math.ceil(max(1, self.placedb.node_info[self.node_name_list[self.num_macro_placed]]['height'] / self.ratio))
        mask = self.get_mask(next_x, next_y)
        next_x_2 = math.ceil(max(1, self.placedb.node_info[self.node_name_list[self.num_macro_placed+1]]['width'] / self.ratio))
        next_y_2 = math.ceil(max(1, self.placedb.node_info[self.node_name_list[self.num_macro_placed+1]]['height'] / self.ratio))
        mask_2 = self.get_mask(next_x_2, next_y_2)
        for net_name in self.placedb.net_info:
            self.net_placed_set[net_name] = set()

        # 测试不加入next node信息时算法的表现
        if self.ignore_next:
            print("ignore next node")
            net_img_2 = np.zeros((self.grid, self.grid))
            mask_2 = np.ones((self.grid, self.grid))
        self.state = np.concatenate((np.array([self.num_macro_placed]), canvas.flatten(), 
            net_img.flatten(), mask.flatten(), net_img_2.flatten(), mask_2.flatten(), 
            np.array([next_x/self.grid, next_y/self.grid])), axis = 0)
        self.node_x_max = 0
        self.node_x_min = self.grid
        self.node_y_max = 0
        self.node_y_min = self.grid

        return self.state

    def save_fig(self, file_path: Path):
        """
        Save the current placement of nodes as a figure.

        Parameters:
        file_path (str): The path where the figure will be saved.
        """
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111, aspect='equal')
        ax1.axes.xaxis.set_visible(False)
        ax1.axes.yaxis.set_visible(False)
        for node_name in self.node_pos:
            x, y, size_x, size_y = self.node_pos[node_name]
            facecolor = 'blue'
            if hasattr(self.placedb, 'hard_macro_info') and node_name not in self.placedb.hard_macro_info:
                facecolor = 'red'
            ax1.add_patch(
                patches.Rectangle(
                    (x/self.grid, y/self.grid),   # (x,y)
                    size_x/self.grid,          # width
                    size_y/self.grid, linewidth=1, edgecolor='k',
                    facecolor=facecolor
                )
            )
        fig1.savefig(file_path, dpi=90, bbox_inches='tight')
        plt.close()

    @timeit
    def save_flyline(self, file_path: Path) -> None:
        # 参数配置
        jitter = 3          # 基础抖动幅度
        base_strength = 20  # 基础弯曲强度
        line_alpha = 0.5    # 线条透明度
        line_width = 0.8    # 线条宽度
        ref_distance = 150  # 弧度计算参考距离

        def generate_distinct_color(n1, n2):
            """生成高区分度的RGB颜色"""
            # 创建排序后的节点对标识
            node_pair = "".join(sorted([n1, n2]))
            # 生成哈希值并映射到色相空间
            hue = int(hashlib.md5(node_pair.encode()).hexdigest()[:5], 16) % 360
            # 转换HSV到RGB（固定饱和度80%，明度90%）
            rgb = colorsys.hsv_to_rgb(hue/360.0, 0.8, 0.9)
            return (rgb[0], rgb[1], rgb[2])  # 返回0-1范围的RGB元组

        def calculate_control_points(x1, y1, x2, y2):
            """计算带方向控制的贝塞尔曲线控制点"""
            dx = x2 - x1
            dy = y2 - y1
            distance = np.hypot(dx, dy)
            
            if distance == 0:
                return (x1, y1), (x2, y2)
            
            # 计算统一方向（始终向右凸起）
            dir_x = dy / distance  # 垂直于连线方向的单位向量
            dir_y = -dx / distance
            
            # 动态弯曲强度：距离越远弯曲越大
            strength = base_strength * (distance / ref_distance) ** 0.5
            
            # 控制点偏移计算
            cp_offset_x = dir_x * strength
            cp_offset_y = dir_y * strength
            
            # 设置控制点位置
            cp1 = (x1 + dx*0.25 + cp_offset_x, 
                y1 + dy*0.25 + cp_offset_y)
            cp2 = (x1 + dx*0.75 + cp_offset_x, 
                y1 + dy*0.75 + cp_offset_y)
            
            return cp1, cp2


        # # 计算节点中心坐标
        # node_centers = {}
        # for node_id, info in node_infos.items():
        #     if node_id.startswith('o'):
        #         x_center = info[0] + info[2] / 2
        #         y_center = info[1] + info[3] / 2
        #         node_centers[node_id] = (x_center, y_center)

        # 创建画布
        _, ax = plt.subplots(figsize=(20, 20))

        node2id_dict = {s:i+1 for i,s in enumerate(self.node_name_list)}

        for node_name in self.node_pos:
            x, y, size_x, size_y = self.node_pos[node_name]
            facecolor = 'cyan'
            if hasattr(self.placedb, 'hard_macro_info') and node_name not in self.placedb.hard_macro_info:
                facecolor = 'red'
            
            x = round(x*self.ratio)
            y = round(y*self.ratio)
            size_x = round(size_x * self.ratio)
            size_y = round(size_y * self.ratio)
            ax.add_patch(
                patches.Rectangle(
                    (x, y),   # (x,y)
                    size_x,   # width
                    size_y,   # height
                    linewidth=1, edgecolor='k',facecolor=facecolor
                )
            )
            ax.text(x+size_x/2, y+size_y/2, node2id_dict[node_name],
                    ha='center', va='center', fontsize=8, color='darkblue')
            
        # 设置画布属性
        ax.autoscale()
        ax.set_aspect('equal')
        plt.grid(True, linestyle=':', alpha=0.3)
        plt.title("Color-Coded Network Visualization", fontsize=14)
        plt.xlabel("X Coordinate", fontsize=10)
        plt.ylabel("Y Coordinate", fontsize=10)

        # 保存图像
        plt.savefig(file_path, bbox_inches='tight', dpi=300)

        all_connections = []
        for net_name in self.placedb.net_info:
            points = []
            if os.getenv('PLACEENV_IGNORE_PORT', '0') != '1':
                for port_info in self.placedb.net_info[net_name]['ports'].values():
                    name = port_info['key']
                    x, y = port_info['pin_offset']
                    points.append((name, x, y))
            for node_info in self.placedb.net_info[net_name]['nodes'].values():
                name = node_info['key']
                x, y, _, _ = self.node_pos[name]
                dx, dy = node_info['pin_offset']
                points.append((name, round(x*self.ratio+dx), round(y*self.ratio+dy)))

            for p1, p2 in combinations(points, 2):
                # 采样10%的边
                if random.random() > 0.01:
                    continue
                n1, x1, y1 = p1
                n2, x2, y2 = p2

                # 计算控制点
                cp1, cp2 = calculate_control_points(x1, y1, x2, y2)
                
                # 生成颜色
                color = generate_distinct_color(n1, n2)
                
                # 计算连线长度用于排序
                length = np.hypot(x2-x1, y2-y1)
                
                all_connections.append((
                    length,
                    [(x1, y1), cp1, cp2, (x2, y2)],
                    color
                ))

        # 按连线长度降序排序（先画长线）
        all_connections.sort(reverse=True, key=lambda x: x[0])

        # 绘制所有连线
        for length, points, color in all_connections:
            path = Path(
                np.array([points[0], points[1], points[2], points[3]]),
                [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
            )
            patch = patches.PathPatch(
                path,
                facecolor='none',
                edgecolor=color,
                linewidth=line_width,
                alpha=line_alpha,
                capstyle='round'
            )
            ax.add_patch(patch)

        # 保存飞线图
        flyline_fig_name = file_path.with_suffix('').as_posix() + '_flyline.png'
        plt.savefig(flyline_fig_name, bbox_inches='tight', dpi=300)
        plt.close()


    def save_pl_file(self, file_path):
        """
        save hard macro placement resut in .pl file
        """
        with open(file_path, 'w') as fwrite:
            node_place = {}
            for node_name in self.node_pos:
                if hasattr(self.placedb, 'hard_macro_info') and node_name not in self.placedb.hard_macro_info:
                    continue
                x, y, _, _ = self.node_pos[node_name]
                x = round((x+1) * self.ratio)
                y = round((y+1) * self.ratio)
                node_place[node_name] = (x, y)
            print(f"Node Num for writing in .pl file: {len(node_place)}")

            for node_name in node_place:
                fwrite.write(f"{node_name}\t{x}\t{y}\t:\tN /FIXED\n")
            print("placement has been saved to {}".format(file_path))
        

    
    def get_net_img(self, is_next_next = False):
        """
            WireMask
        """
        net_img = np.zeros((self.grid, self.grid))
        if not is_next_next:
            next_node_name = self.node_name_list[self.num_macro_placed]
        elif self.num_macro_placed + 1 < len(self.node_name_list):
            next_node_name = self.node_name_list[self.num_macro_placed + 1]
        else:
            return net_img

        for net_name in self.placedb.node2net_dict[next_node_name]:
            if net_name in self.net_min_max_ord:
                delta_pin_x = round(self.placedb.net_info[net_name]["nodes"][next_node_name]["pin_offset"][0]/self.ratio)
                delta_pin_y = round(self.placedb.net_info[net_name]["nodes"][next_node_name]["pin_offset"][1]/self.ratio)

                start_x = self.net_min_max_ord[net_name]['min_x'] - delta_pin_x
                end_x = self.net_min_max_ord[net_name]['max_x'] - delta_pin_x
                start_y = self.net_min_max_ord[net_name]['min_y'] - delta_pin_y
                end_y = self.net_min_max_ord[net_name]['max_y'] - delta_pin_y
                start_x = min(start_x, self.grid)
                start_y = min(start_y, self.grid)
                if not 'weight' in self.placedb.net_info[net_name]:
                    weight = 1.0
                else:
                    weight = self.placedb.net_info[net_name]['weight']
                for i in range(0, start_x):
                    net_img[i, :] += (start_x - i) * weight
                for i in range(end_x+1, self.grid):
                    net_img[i, :] +=  (i- end_x) * weight
                for j in range(0, start_y):
                    net_img[:, j] += (start_y - j) * weight
                for j in range(end_y+1, self.grid):
                    net_img[:, j] += (j - start_y) * weight
        return net_img

    def step(self, action):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg

        canvas = self.state[1: 1+self.grid*self.grid].reshape(self.grid, self.grid)
        mask = self.state[1+self.grid*self.grid*2: 1+self.grid*self.grid*3].reshape(self.grid, self.grid)
        reward = 0
        x = round(action // self.grid)
        y = round(action % self.grid)
    
        if mask[x][y] == 1:
            reward += -200000
                
        node_name = self.node_name_list[self.num_macro_placed]
        size_x = math.ceil(max(1, self.placedb.node_info[node_name]['width']/self.ratio))
        size_y = math.ceil(max(1, self.placedb.node_info[node_name]['height']/self.ratio))

        assert abs(size_x - self.state[-2]*self.grid) < 1e-5
        assert abs(size_y - self.state[-1]*self.grid) < 1e-5

        canvas[x : x+size_x, y : y+size_y] = 1.0
        canvas[x : x + size_x, y] = 0.5
        if y + size_y -1 < self.grid:
            canvas[x : x + size_x, max(0, y + size_y -1)] = 0.5
        canvas[x, y: y + size_y] = 0.5
        if x + size_x - 1 < self.grid:
            canvas[max(0, x+size_x-1), y: y + size_y] = 0.5
        self.node_pos[self.node_name_list[self.num_macro_placed]] = (x, y, size_x, size_y)

        for net_name in self.placedb.node2net_dict[node_name]:
            self.net_placed_set[net_name].add(node_name)

            pin_offset_x, pin_offset_y = self.placedb.net_info[net_name]["nodes"][node_name]["pin_offset"]
            pin_x = round((x * self.ratio + pin_offset_x) / self.ratio)
            pin_y = round((y * self.ratio + pin_offset_y) / self.ratio)

            if net_name in self.net_min_max_ord:
                start_x = self.net_min_max_ord[net_name]['min_x']
                end_x = self.net_min_max_ord[net_name]['max_x']
                start_y = self.net_min_max_ord[net_name]['min_y']
                end_y = self.net_min_max_ord[net_name]['max_y']
                delta_x = end_x - start_x
                delta_y = end_y - start_y
                if delta_x > 0 or delta_y > 0:
                    self.rudy[start_x : end_x +1, start_y: end_y +1] -= 1/(delta_x+1) + 1/(delta_y+1) 
                weight = 1.0
                if 'weight' in self.placedb.net_info[net_name]:
                    weight = self.placedb.net_info[net_name]['weight']
 
                if pin_x > self.net_min_max_ord[net_name]['max_x']:
                    reward += weight * (self.net_min_max_ord[net_name]['max_x'] - pin_x)
                    self.net_min_max_ord[net_name]['max_x'] = pin_x
                    self.net_fea[self.placedb.net_info[net_name]['id']][1] = pin_x / self.grid
                elif pin_x < self.net_min_max_ord[net_name]['min_x']:
                    reward += weight * (pin_x - self.net_min_max_ord[net_name]['min_x'])
                    self.net_min_max_ord[net_name]['min_x'] = pin_x
                    self.net_fea[self.placedb.net_info[net_name]['id']][0] = pin_x / self.grid
                if pin_y > self.net_min_max_ord[net_name]['max_y']:
                    reward += weight * (self.net_min_max_ord[net_name]['max_y'] - pin_y)
                    self.net_min_max_ord[net_name]['max_y'] = pin_y
                    self.net_fea[self.placedb.net_info[net_name]['id']][3] = pin_y / self.grid
                elif pin_y < self.net_min_max_ord[net_name]['min_y']:
                    reward += weight * (pin_y - self.net_min_max_ord[net_name]['min_y'])
                    self.net_min_max_ord[net_name]['min_y'] = pin_y
                    self.net_fea[self.placedb.net_info[net_name]['id']][2] = pin_y / self.grid
                start_x = self.net_min_max_ord[net_name]['min_x']
                end_x = self.net_min_max_ord[net_name]['max_x']
                start_y = self.net_min_max_ord[net_name]['min_y']
                end_y = self.net_min_max_ord[net_name]['max_y']
                delta_x = end_x - start_x
                delta_y = end_y - start_y
                self.rudy[start_x : end_x +1, start_y: end_y +1] += 1/(delta_x+1) + 1/(delta_y+1) 
            else:
                self.net_min_max_ord[net_name] = {}
                self.net_min_max_ord[net_name]['max_x'] = pin_x
                self.net_min_max_ord[net_name]['min_x'] = pin_x
                self.net_min_max_ord[net_name]['max_y'] = pin_y
                self.net_min_max_ord[net_name]['min_y'] = pin_y
                start_x = self.net_min_max_ord[net_name]['min_x']
                end_x = self.net_min_max_ord[net_name]['max_x']
                start_y = self.net_min_max_ord[net_name]['min_y']
                end_y = self.net_min_max_ord[net_name]['max_y']
                self.net_fea[self.placedb.net_info[net_name]['id']][1] = pin_x / self.grid
                self.net_fea[self.placedb.net_info[net_name]['id']][0] = pin_x / self.grid
                self.net_fea[self.placedb.net_info[net_name]['id']][3] = pin_y / self.grid
                self.net_fea[self.placedb.net_info[net_name]['id']][2] = pin_y / self.grid
                reward += 0

        self.num_macro_placed += 1
        net_img = np.zeros((self.grid, self.grid))
        net_img_2 = np.zeros((self.grid, self.grid))

        if self.num_macro_placed < self.placed_num_macro:
            net_img = self.get_net_img()
            net_img_2 = self.get_net_img(is_next_next= True)
            if net_img.max() >0 or net_img_2.max()>0:
                net_img /= max(net_img.max(), net_img_2.max())
                net_img_2 /= max(net_img.max(), net_img_2.max())

        if self.node_x_max < x:
            self.node_x_max = x
        if self.node_x_min > x:
            self.node_x_min = x
        if self.node_y_max < y:
            self.node_y_max = y
        if self.node_y_min > y:
            self.node_y_min = y

        if self.num_macro_placed == self.num_macro or \
            (self.placed_num_macro is not None and self.num_macro_placed == self.placed_num_macro): 
            done = True
        else:
            done = False
        mask = np.ones((self.grid, self.grid))
        mask_2 = np.ones((self.grid, self.grid))
        if not done: # get next macro size and pre-mask the solution
            next_x = math.ceil(max(1, self.placedb.node_info[self.node_name_list[self.num_macro_placed]]['width']/self.ratio))
            next_y = math.ceil(max(1, self.placedb.node_info[self.node_name_list[self.num_macro_placed]]['height']/self.ratio))
            mask = self.get_mask(next_x, next_y)
            if self.num_macro_placed + 1 < self.placed_num_macro:
                next_x_2 = math.ceil(max(1, self.placedb.node_info[self.node_name_list[self.num_macro_placed+1]]['width']/self.ratio))
                next_y_2 = math.ceil(max(1, self.placedb.node_info[self.node_name_list[self.num_macro_placed+1]]['height']/self.ratio))
                mask_2 = self.get_mask(next_x_2, next_y_2)
        else:
            next_x = 0
            next_y = 0

        # 测试不加入next node信息时算法的表现
        if self.ignore_next:
            net_img_2 = np.zeros((self.grid, self.grid))
            mask_2 = np.ones((self.grid, self.grid))

        self.state = np.concatenate((np.array([self.num_macro_placed]), canvas.flatten(), 
            net_img.flatten(), mask.flatten(), net_img_2.flatten(), mask_2.flatten(),
            np.array([next_x/self.grid, next_y/self.grid])), axis = 0)
        return self.state, reward, done, {"raw_reward": reward, "net_img": net_img, "mask": mask}
    
    
    def get_mask(self, next_x, next_y):
        """
            PositionMask
        """
        mask = np.zeros((self.grid, self.grid))
        for node_name in self.node_pos:
            startx = max(0, self.node_pos[node_name][0] - next_x + 1)
            starty = max(0, self.node_pos[node_name][1] - next_y + 1)
            endx = min(self.node_pos[node_name][0] + self.node_pos[node_name][2] - 1, self.grid - 1)
            endy = min(self.node_pos[node_name][1] + self.node_pos[node_name][3] - 1, self.grid - 1)
            mask[startx: endx + 1, starty : endy + 1] = 1
        mask[self.grid - next_x + 1:,:] = 1
        mask[:, self.grid - next_y + 1:] = 1
        return mask

if __name__ == "__main__":
    placedb = LefDefPlaceDB()
    env = PlaceEnv(placedb)
    print(env.observation_space)
    print(env.action_space)
    env.reset()
    done = False
    import time
    t0 = time.time()
    index = 0
    while not done:
        # 读取def中的位置信息进行摆放
        macro_name = env.node_name_list[index]
        coordinate = env.placedb.place_instance_dict[macro_name]['coordinate']
        x = math.ceil(coordinate[0] / env.ratio)
        y = math.ceil(coordinate[1] / env.ratio)

        action = x * env.grid + y
        state, reward, done, info = env.step(action)
        index += 1
    
    print(f"Time used: {time.time() - t0}s")
    env.save_fig("./test.png")