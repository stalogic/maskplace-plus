import re
import os
import gzip
from pathlib import Path

from util import is_gzip_file


class LefDefReader(object):
    """
    Read LEF and DEF files, extract geometric information of cells and pins and save them in lef_dict.
    Returns
        1.lef_dict: {cell_name:{pin:{pin_name:pin_rect}, 'size':[unit*w,unit*h]}}
        2.place_instance_dict: {instance_name:{'type':type, 'coordinate':(x,y), 'size':[w,h], 'orientation':orientation, 'attributes':attributes}}
        3.place_net_dict: {net_name:{'id':id, 'key':net_name, 'nodes':[(node_name, pin_name), ...]}}
        4.place_pin_dict {pin_name:{'direction':direction, 'layer_size':(w,h), 'attributes':attributes, 'coordinate':(x,y), 'orientation':orientation}}
    """

    def __init__(self, data_root:str, design_name:str, cache_root:str, unit:int=2000, **kwargs):
        self.data_root = Path(data_root)
        self.unit = unit
        
        try:
            cache_path = Path(cache_root)
            cache_path.mkdir(parents=True, exist_ok=True)
            self.error_file = open(cache_path / f'{design_name}_lefdef_reader.err', 'w')
        except:
            self.error_file = None

        def_path, lef_list = self._parse_lefdef_files(self.data_root / design_name)

        self.lef_dict = {}
        for lef in lef_list:
            lef_data = self._read_lef(lef, unit)
            self.lef_dict.update(lef_data)

        self.place_instance_dict = {}
        self.place_net_dict = {}
        self.place_pin_dict = {}
        self.place_row_list = []
        self._read_def(def_path)

        if self.error_file is not None:
            self.error_file.close()

    @staticmethod
    def _parse_lefdef_files(design_path:Path) -> tuple[str, list[str]]:
        assert design_path.exists(), f"Design Path '{design_path}' not exists."

        def_path = None
        lef_path = []
        for file_path in design_path.iterdir():
            if file_path.is_file():
                if file_path.name.endswith('lef'):
                    lef_path.append(file_path)
                elif file_path.name.endswith('def'):
                    assert def_path is None, f"Multiple def file is undefined. \n{def_path}\n{file_path}"
                    def_path = file_path
        return def_path, lef_path


    def _read_lef(self, path, unit):
        """ 
        read LEF file, extract geometric information of cells and pins and save them in lef_dict. 

        :param path: path to LEF file.
        :param lef_dict: empty dict or dict from the other LEF.
        :param unit: distance convert factors, e.g., unit = 1000, then 1000 in DEF equals 1 micron in LEF.
        :return: lef_dict {cell_name:{pin:{pin_name:pin_rect}, 'size':[unit*w,unit*h]}}
        """
        lef_dict = {}
        with open(path, 'r') as read_file:
            cell_name = ''
            pin_name = ''
            rect_list_left = []
            rect_list_lower = []
            rect_list_right = []
            rect_list_upper = []
            READ_MACRO = False
            for idx, line in enumerate(read_file):
                if line.lstrip().startswith('MACRO'):
                    READ_MACRO = True
                    cell_name = line.split()[1]
                    lef_dict[cell_name] = {}
                    lef_dict[cell_name]['key'] = cell_name
                    lef_dict[cell_name]['pin'] = {}

                if READ_MACRO:
                    if line.lstrip().startswith('SIZE'):
                        l = re.findall(r'-?\d+\.?\d*e?-?\d*?', line)
                        lef_dict[cell_name]['size'] = [unit * float(l[0]), unit * float(l[1])]  # size [unit*w,unit*h]

                    elif line.lstrip().startswith('PIN'):
                        pin_name = line.split()[1]
                        if pin_name not in lef_dict[cell_name]['pin'].keys():
                            lef_dict[cell_name]['pin'][pin_name] = {}
                            lef_dict[cell_name]['pin'][pin_name]['key'] = pin_name

                    elif line.lstrip().startswith('RECT'):
                        l = line.split()
                        rect_list_left.append(float(l[1]))
                        rect_list_lower.append(float(l[2]))
                        rect_list_right.append(float(l[3]))
                        rect_list_upper.append(float(l[4]))
                    
                    elif line.strip().startswith("DIRECTION"):
                        direction = line.strip().split()[1]
                        lef_dict[cell_name]['pin'][pin_name]['direction'] = direction

                    elif line.lstrip().startswith('END %s\n' % pin_name):
                        rect_left = min(rect_list_left) * unit
                        rect_lower = min(rect_list_lower) * unit
                        rect_right = max(rect_list_right) * unit
                        rect_upper = max(rect_list_upper) * unit

                        if pin_name not in lef_dict[cell_name]['pin'].keys():
                            lef_dict[cell_name]['pin'][pin_name] = {}

                        lef_dict[cell_name]['pin'][pin_name]['rect_left'] = rect_left
                        lef_dict[cell_name]['pin'][pin_name]['rect_lower'] = rect_lower
                        lef_dict[cell_name]['pin'][pin_name]['rect_right'] = rect_right
                        lef_dict[cell_name]['pin'][pin_name]['rect_upper'] = rect_upper

                        rect_list_left = []
                        rect_list_lower = []
                        rect_list_right = []
                        rect_list_upper = []
                    else:
                        print(f"[warning] discard line in macro/pin '{cell_name}/{pin_name}': '[{idx+1}]{line.rstrip()}'", file=self.error_file)
                else:
                    print(f"[warning] discard line: '[{idx+1}]{line.rstrip()}'", file=self.error_file)
        
        return lef_dict

    def _read_def(self, def_path):
        if is_gzip_file(def_path):
            read_file = gzip.open(def_path, "rt")
        else:
            read_file = open(def_path, "r")

        READ_COMPONENTS = False
        READ_NETS = False
        READ_PINS = False
        net = ''

        for line in read_file:
            line = line.lstrip()
            if line.startswith("DIEAREA"):
                die_coordinate = re.findall(r'\d+', line)
                self.die_area = (int(die_coordinate[2]), int(die_coordinate[3]))
            elif line.startswith("ROW CORE_ROW"):
                self._parse_row(line)
            elif line.startswith("COMPONENTS"):
                instance = line.split()
                READ_COMPONENTS = True
                self.num_components = int(instance[1])
            elif line.startswith("END COMPONENTS"):
                READ_COMPONENTS = False
            elif line.startswith('PIN'):
                instance = line.split()
                READ_PINS = True
                self.num_terminal_NIs = int(instance[1])
            elif line.startswith('END PINS'):
                READ_PINS = False
            elif line.startswith("NETS"):
                READ_NETS = True
                instance = line.split()
                self.num_nets = int(instance[1])

            elif line.startswith("END NETS") or line.startswith("SPECIALNETS"):
                READ_NETS = False


            if READ_COMPONENTS:
                if "FIXED" in line:
                    instance = line.split()
                    l = instance.index('(')

                    size = self.lef_dict[instance[2]]['size']
                    name = str(instance[1])
                    type = instance[2]
                    attributes = 'FIXED'
                    coordinate = (int(instance[l + 1]), int(instance[l + 2]))
                    orientation = instance[l + 4]

                    if name not in self.place_instance_dict.keys():
                        self.place_instance_dict[name] = {}

                    self.place_instance_dict[name]['type'] = type
                    self.place_instance_dict[name]['coordinate'] = coordinate
                    self.place_instance_dict[name]['size'] = size
                    self.place_instance_dict[name]['orientation'] = orientation
                    self.place_instance_dict[name]['attributes'] = attributes

                elif "PLACED" in line:
                    instance = line.split()
                    l = instance.index('(')

                    size = self.lef_dict[instance[2]]['size']
                    name = str(instance[1])
                    type = instance[2]
                    attributes = 'PLACED'
                    coordinate = (int(instance[l + 1]), int(instance[l + 2]))
                    orientation = instance[l + 4]

                    if name not in self.place_instance_dict.keys():
                        self.place_instance_dict[name] = {}

                    self.place_instance_dict[name]['type'] = type
                    self.place_instance_dict[name]['coordinate'] = coordinate
                    self.place_instance_dict[name]['size'] = size
                    self.place_instance_dict[name]['orientation'] = orientation
                    self.place_instance_dict[name]['attributes'] = attributes
                else:
                    # ignore the line or raise error
                    if len(line.strip()) < 3 or line.strip().startswith('COMPONENTS') or line.strip().startswith('END COMPONENTS'):
                        pass
                    else:
                        raise ValueError(f"unknown line in COMPONENTS: {line}")

            if READ_PINS:
                stripline = line.strip()
                if line.startswith('-'):
                    instance = line.split()
                    name = instance[1]
                    direction = instance[7]

                    if name not in self.place_pin_dict.keys():
                        self.place_pin_dict[name] = {}
                    self.place_pin_dict[name]['direction'] = direction

                elif stripline.startswith('+ LAYER'):
                    instance = line.split()
                    l = instance.index('(')

                    layer_size = (int(instance[l + 1]), int(instance[l + 2]), int(instance[l + 5]), int(instance[l + 6]))

                    if name not in self.place_pin_dict.keys():
                        self.place_pin_dict[name] = {}
                    self.place_pin_dict[name]['layer_size'] = layer_size
                elif stripline.startswith('+ FIXED') or stripline.startswith('+ PLACED'):
                    instance = line.split()
                    l = instance.index('(')

                    attributes = instance[1]
                    coordinate = (int(instance[l + 1]), int(instance[l + 2]))
                    orientation = instance[l + 4]

                    if name not in self.place_pin_dict.keys():
                        self.place_pin_dict[name] = {}
                    self.place_pin_dict[name]['attributes'] = attributes
                    self.place_pin_dict[name]['coordinate'] = coordinate
                    self.place_pin_dict[name]['orientation'] = orientation

            if READ_NETS:  # get route_net_dict
                if line.startswith('-'):
                    net = line.split()[1]
                    if net not in self.place_net_dict.keys():
                        net_id = len(self.place_net_dict)
                        self.place_net_dict[net] = {'id': net_id, 'key': net, 'nodes': []}

                elif line.startswith('('):  # get pin names in each net
                    # if net not in self.place_net_dict.keys():
                    #     self.place_net_dict[net] = []

                    l = line.split()
                    n = 0
                    for k in l:
                        if k == '(':
                            self.place_net_dict[net]['nodes'].append([l[n + 1], l[n + 2]])
                        n += 1
                else:
                    if len(line.strip()) < 3 or line.strip().startswith('NETS') \
                        or line.strip().startswith('END NETS') \
                        or line.strip().startswith('+ USE POWER') \
                        or line.strip().startswith('+ USE GROUND'):
                        pass
                    else:
                        raise ValueError(f"unknown line in NETS: {line}")

        self.num_physical_nodes = self.num_components + self.num_terminal_NIs
        read_file.close()

    def _parse_row(self, line:str):
        """
        parse row information from DEF file.
        example: ROW CORE_ROW_0 FreePDK45_38x28_10R_NP_162NW_34O 0 0 FS DO 8421 BY 1 STEP 380 0
        """
        parts = line.split()

        row_item = {}
        # 提取子行原点坐标信息并转换为整数
        row_item["SubrowOrigin"] = int(parts[3])
        # 提取坐标信息并转换为整数
        row_item["Coordinate"] = int(parts[4])
        # 提取站点宽度信息，并同时赋值给站点间距（假设它们相等），转换为整数
        row_item["Sitewidth"] = row_item["Sitespacing"] = int(parts[11])
        # 提取站点数量信息并转换为整数
        row_item["NumSites"] = int(parts[7])
        # 根据站点方向（"FS"为0，其他为1）设置站点方向信息
        row_item["Siteorient"] = 0 if parts[5] == "FS" else 1
        # 设置固定的高度值
        row_item["Height"] = 2800
        # 设置站点对称性信息
        row_item["Sitesymmetry"] = 1

        self.place_row_list.append(row_item)
