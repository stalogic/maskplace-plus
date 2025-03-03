import argparse
from collections import namedtuple

import os
import numpy as np
import matplotlib.pyplot as plt
import pathlib

import time
from tqdm import tqdm
import random
import gym
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

import place_env

from placedb import LefDefPlaceDB, LefDefReader, convert_to_soft_macro_placedb

from comp_res import comp_res



# Parameters
parser = argparse.ArgumentParser(description='Solve Macro Placement Task with PPO')
# design data parameters
parser.add_argument('--data_root', default='./benchmark/', help='the parent dir of innovus workspace')
parser.add_argument('--design_name', default='ariane133', help='the parent dir of design_name')
parser.add_argument('--cache_root', default='./cache', help='save path')
parser.add_argument('--unit', default=2000, help='unit defined in the begining of DEF')

# training parameters
parser.add_argument('--gamma', type=float, default=0.95, metavar='G', help='discount factor (default: 0.9)')
parser.add_argument('--seed', type=int, default=42, metavar='N', help='random seed (default: 0)')
parser.add_argument('--disable_tqdm', type=int, default=1)
parser.add_argument('--lr', type=float, default=2.5e-3)
parser.add_argument('--soft_coefficient', type=float, default = 1)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--result_root', type=str, default='./result/', help='path for save result')
parser.add_argument('--log_root', type=str, default='./logs/', help='path for save result')
parser.add_argument('--cuda', type=str, default='', help='cuda device for set CUDA_VISIBLE_DEVICES')
parser.add_argument('--ignore_next', action='store_true', default=False, help='ignore next macro')
parser.add_argument('--debug', action='store_true', default=False, help='debug mode')
args = parser.parse_args()

# set device to cpu or cuda
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")

benchmark = args.design_name
result_root = args.result_root
log_root = args.log_root
lefdef_reader = LefDefReader(**vars(args))
placedb = LefDefPlaceDB(place_net_dict=lefdef_reader.place_net_dict, 
                        place_instance_dict=lefdef_reader.place_instance_dict, 
                        place_pin_dict=lefdef_reader.place_pin_dict, 
                        lef_dict=lefdef_reader.lef_dict,
                        die_area=lefdef_reader.die_area)

placedb = convert_to_soft_macro_placedb(placedb=placedb, parser=lefdef_reader)

grid = 224
placed_num_macro = len(placedb.macro_info)
env = gym.make('place_env-v0', placedb = placedb, grid = grid, ignore_next = args.ignore_next).unwrapped

num_emb_state = 64 + 2 + 1
num_state = 1 + grid*grid*5 + 2

def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    # env.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

num_action = env.action_space.shape
seed_torch(args.seed)

Transition = namedtuple('Transition',['state', 'action', 'reward', 'a_log_prob', 'next_state', 'reward_intrinsic'])
TrainingRecord = namedtuple('TrainRecord',['episode', 'reward'])
print(f"seed = {args.seed}")
print(f"lr = {args.lr}")
print(f"placed_num_macro = {placed_num_macro}")


class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 8, 1),
            nn.ReLU(),
            nn.Conv2d(8, 8, 1),
            nn.ReLU(),
            nn.Conv2d(8, 1, 1),
        )
    def forward(self, x):
        return self.cnn(x)


class MyCNNCoarse(nn.Module):
    def __init__(self, res_net):
        super(MyCNNCoarse, self).__init__()
        self.cnn = res_net.to(device)
        self.cnn.fc = torch.nn.Linear(512, 16*7*7)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding = 1), #14
            nn.ReLU(),
            nn.ConvTranspose2d(8, 4, 3, stride=2, padding=1, output_padding = 1), #28
            nn.ReLU(),
            nn.ConvTranspose2d(4, 2, 3, stride=2, padding=1, output_padding = 1), #56
            nn.ReLU(),
            nn.ConvTranspose2d(2, 1, 3, stride=2, padding=1, output_padding = 1), #112
            nn.ReLU(),
            nn.ConvTranspose2d(1, 1, 3, stride=2, padding=1, output_padding = 1), #224
        )
    def forward(self, x):
        x = self.cnn(x).reshape(-1, 16, 7, 7)
        return self.deconv(x)


class Actor(nn.Module):
    def __init__(self, cnn, gcn, cnn_coarse):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_emb_state, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, grid * grid)
        self.cnn = cnn
        self.cnn_coarse = cnn_coarse
        self.gcn = None
        self.softmax = nn.Softmax(dim=-1)
        self.merge = nn.Conv2d(2, 1, 1)

    def forward(self, x, graph = None, cnn_res = None, gcn_res = None, graph_node = None):
        if not cnn_res:
            cnn_input = x[:, 1+grid*grid*1: 1+grid*grid*5].reshape(-1, 4, grid, grid)
            mask = x[:, 1+grid*grid*2: 1+grid*grid*3].reshape(-1, grid, grid)
            mask = mask.flatten(start_dim=1, end_dim=2)
            cnn_res = self.cnn(cnn_input)
            coarse_input = torch.cat((x[:, 1: 1+grid*grid*2].reshape(-1, 2, grid, grid),
                                        x[:, 1+grid*grid*3: 1+grid*grid*4].reshape(-1, 1, grid, grid)
                                        ),dim= 1).reshape(-1, 3, grid, grid)
            cnn_coarse_res = self.cnn_coarse(coarse_input)
            cnn_res = self.merge(torch.cat((cnn_res, cnn_coarse_res), dim=1))
        net_img = x[:, 1+grid*grid: 1+grid*grid*2]
        net_img = net_img + x[:, 1+grid*grid*2: 1+grid*grid*3] * 10
        net_img_min = net_img.min() + args.soft_coefficient
        mask2 = net_img.le(net_img_min).logical_not().float()

        x = cnn_res
        x = x.reshape(-1, grid * grid)
        x = torch.where(mask + mask2 >=1.0, -1.0e10, x.double())
        x = self.softmax(x)

        return x, cnn_res, gcn_res


class Critic(nn.Module):
    def __init__(self, cnn, gcn, cnn_coarse, res_net):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 64)
        self.state_value = nn.Linear(64, 1)
        self.pos_emb = nn.Embedding(1400, 64)
        self.cnn = cnn
        self.gcn = gcn
    def forward(self, x, graph = None, cnn_res = None, gcn_res = None, graph_node = None):
        x1 = F.relu(self.fc1(self.pos_emb(x[:, 0].long())))
        x2 = F.relu(self.fc2(x1))
        value = self.state_value(x2)
        return value


class PPO():
    clip_param = 0.2
    max_grad_norm = 0.5
    ppo_epoch = 10
    if placed_num_macro:
        buffer_capacity = 5 * (placed_num_macro)
    else:
        buffer_capacity = 5120
    batch_size = args.batch_size
    print("batch_size = {}".format(batch_size))

    def __init__(self):
        super(PPO, self).__init__()
        self.gcn = None
        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.cnn = MyCNN().to(device)
        self.cnn_coarse = MyCNNCoarse(self.resnet).to(device)
        self.actor_net = Actor(cnn = self.cnn, gcn = self.gcn, cnn_coarse = self.cnn_coarse).float().to(device)
        self.critic_net = Critic(cnn = self.cnn, gcn = self.gcn,  cnn_coarse = None, res_net = self.resnet).float().to(device)
        self.buffer = []
        self.counter = 0
        self.training_step = 0
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), args.lr)
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), args.lr)

    def load_param(self, path):
        checkpoint = torch.load(path, map_location=torch.device(device))
        self.actor_net.load_state_dict(checkpoint['actor_net_dict'])
        self.critic_net.load_state_dict(checkpoint['critic_net_dict'])
    
    def select_action(self, state):
        state = torch.from_numpy(state).float().to(device).unsqueeze(0)
        with torch.no_grad():
            action_probs, _, _ = self.actor_net(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)
        return action.item(), action_log_prob.item()

    def get_value(self, state):
        state = torch.from_numpy(state)
        with torch.no_grad():
            value = self.critic_net(state)
        return value.item()

    def save_param(self, save_flag):
        torch.save({"actor_net_dict": self.actor_net.state_dict(),
                    "critic_net_dict": self.critic_net.state_dict()},
                    f"{save_flag}_net_dict.pkl")

    def store_transition(self, transition):
        self.buffer.append(transition)
        self.counter+=1
        return self.counter % self.buffer_capacity == 0

    def update(self):
        state = torch.tensor(np.array([t.state for t in self.buffer]), dtype=torch.float)
        action = torch.tensor(np.array([t.action for t in self.buffer]), dtype=torch.float).view(-1, 1).to(device)
        reward = torch.tensor(np.array([t.reward for t in self.buffer]), dtype=torch.float).view(-1, 1).to(device)
        old_action_log_prob = torch.tensor(np.array([t.a_log_prob for t in self.buffer]), dtype=torch.float).view(-1, 1).to(device)
        del self.buffer[:]
        target_list = []
        target = 0
        for i in range(reward.shape[0]-1, -1, -1):
            if state[i, 0] >= placed_num_macro - 1:
                target = 0
            r = reward[i, 0].item()
            target = r + args.gamma * target
            target_list.append(target)
        target_list.reverse()
        target_v_all = torch.tensor(np.array([t for t in target_list]), dtype=torch.float).view(-1, 1).to(device)
       
        for _ in range(self.ppo_epoch): # iteration ppo_epoch 
            for index in tqdm(BatchSampler(SubsetRandomSampler(range(self.buffer_capacity)), self.batch_size, True),
                disable = args.disable_tqdm):
                self.training_step +=1
                
                action_probs, _, _ = self.actor_net(state[index].to(device))
                dist = Categorical(action_probs)
                action_log_prob = dist.log_prob(action[index].squeeze())
                ratio = torch.exp(action_log_prob - old_action_log_prob[index].squeeze())
                target_v = target_v_all[index]                
                critic_net_output = self.critic_net(state[index].to(device))
                advantage = (target_v - critic_net_output).detach()

                L1 = ratio * advantage.squeeze() 
                L2 = torch.clamp(ratio, 1-self.clip_param, 1+self.clip_param) * advantage.squeeze() 
                action_loss = -torch.min(L1, L2).mean() # MAX->MIN desent

                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                value_loss = F.smooth_l1_loss(self.critic_net(state[index].to(device)), target_v)
                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()

                # writer.add_scalar('action_loss', action_loss, self.training_step)
                # writer.add_scalar('value_loss', value_loss, self.training_step)



def main():

    strftime = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    RUN_ID = f"{benchmark}_{strftime}_seed_{args.seed}_pnm_{placed_num_macro}"
    RESULT_PATH = pathlib.Path(result_root) / RUN_ID
    LOG_PATH = pathlib.Path(log_root)
    LOG_PATH.mkdir(parents=True, exist_ok=True)

    FIGURE_PATH = RESULT_PATH / 'figures'
    MODEL_PATH = RESULT_PATH / 'saved_model'
    PLACE_PATH = RESULT_PATH / 'placement'

    FIGURE_PATH.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    PLACE_PATH.mkdir(parents=True, exist_ok=True)
    
    log_file_name = LOG_PATH / f"{RUN_ID}.csv"
    fwrite = open(log_file_name, "w")

    agent = PPO()
    load_model_path = None
    if load_model_path:
       agent.load_param(load_model_path)

    training_records = []
    running_reward = -float('inf')
    best_reward = running_reward

    for i_epoch in range(2000):
        score = 0
        raw_score = 0
        start = time.time()
        state = env.reset()

        done = False
        while done is False:
            state_tmp = state.copy()
            action, action_log_prob = agent.select_action(state)
        
            next_state, reward, done, info = env.step(action)
            assert next_state.shape == (num_state, )
            reward_intrinsic = 0
            trans = Transition(state_tmp, action, reward / 200.0, action_log_prob, next_state, reward_intrinsic)
            if agent.store_transition(trans):                
                assert done == True
                agent.update()
            score += reward
            raw_score += info["raw_reward"]
            state = next_state
        end = time.time()

        print(f"Game Time: {end - start:.2f}s")

        if i_epoch == 0:
            running_reward = score
        running_reward = running_reward * 0.9 + score * 0.1
        print(f"score = {score:.3e}, raw_score = {raw_score:.3e}")

        if running_reward > best_reward * 0.98 or args.debug:
            best_reward = running_reward
            if i_epoch > 10 or args.debug:
                try:
                    print("start try")
                    # cost is the routing estimation based on the MST algorithm
                    hpwl, cost = comp_res(placedb, env.node_pos, env.ratio)
                    print(f"hpwl = {hpwl:.3e}\tcost = {cost:.3e}")
                    
                    strftime_now = time.strftime("%Y%m%d-%H-%M-%S", time.localtime())
                    save_flag = f"{strftime_now}_I[{i_epoch}]_S[{int(raw_score)}]_H[{hpwl:.3e}]_C[{cost:.3e}]"

                    agent.save_param(MODEL_PATH / save_flag)
                    fig_name = FIGURE_PATH / f"{save_flag}.png"
                    env.save_flyline(fig_name)
                    print(f"save_figure: {fig_name}")
                    pl_name = PLACE_PATH / f"{save_flag}.pl"
                    env.save_pl_file(pl_name)
                except:
                    assert False
        
        training_records.append(TrainingRecord(i_epoch, running_reward))
        if i_epoch % 1 ==0:
            print(f"Epoch {i_epoch}, Moving average score is: {running_reward:.3e} Best reward: {best_reward:.3e}")
            fwrite.write(f"{i_epoch},{score:.3e},{running_reward:.3e},{best_reward:.3e},{agent.training_step}\n")
            fwrite.flush()
        if running_reward > -100:
            print("Solved! Moving average score is now {}!".format(running_reward))
            env.close()
            agent.save_param()
            break

if __name__ == '__main__':
    main()
