from model import G_DQN, ReplayBuffer
import numpy as np
import random
import torch
import torch.nn as nn
from torch.optim import Adam
from arguments import args

entire_state = (65, 65, 3)
predator1_obs = (10, 10, 3)
predator2_obs = (6, 6, 3)
dim_act = 13
n_predator1 = 10
n_predator2 = 10
eps_decay = 0.1
#batch_size = 10
predator1_adj = (256,256)
predator2_adj = (36,36)

import argparse



class MADQN():  # def __init__(self,  dim_act, observation_state):
    def __init__(self, n_predator1, n_predator2, predator1_obs, predator2_obs, dim_act , entire_state, shared, device = 'cpu', buffer_size = 500):
        self.entire_state = entire_state
        self.predator1_obs = predator1_obs
        self.predator2_obs = predator2_obs
        self.n_predator1 = n_predator1
        self.n_predator2 = n_predator2
        self.dim_act = dim_act
        self.epsilon = args.eps
        self.eps_decay = args.eps_decay
        self.device = device

        # ?? n_predator1 ?? predator1? dqn ??, ? ?? ?? predator2 ? dqn ?? observation ? ??? ??? ?? dqn? ???? ??.
        self.gdqns = [G_DQN(self.dim_act, self.predator1_obs).to(self.device) for _ in range(self.n_predator1)] + [
            G_DQN(self.dim_act, self.predator2_obs).to(self.device) for _ in range(self.n_predator2)]
        self.gdqn_targets = [G_DQN(self.dim_act, self.predator1_obs).to(self.device) for _ in range(self.n_predator1)] + [
            G_DQN(self.dim_act, self.predator2_obs).to(self.device) for _ in range(self.n_predator2)]  # ??? ??? ?? target dqn ??
        self.buffers = [ReplayBuffer(capacity=buffer_size) for _ in range(self.n_predator1 + self.n_predator2)]
        self.gdqn_optimizers = [Adam(x.parameters(), lr=0.001) for x in self.gdqns]
        #self.target_optimizer = [Adam(x.parameters(), lr=0.001) for x in self.gdqns_target] #?? ????? ?? ??? weight???? ???? ????
        self.criterion = nn.MSELoss()

        # shared gnn ? ??!
        self.shared = shared
        self.pos = None
        self.view_range = None

        self.adj = None
        self.idx = None

        self.gdqn = None
        self.gdqn_target = None

        self.gdqn_optimizer = None
        self.target_optimizer = None

        self.buffer = None




    def target_update(self):  # ????? target ???? ? tensorflow ?????
        weights = self.gdqn.state_dict()  # behavior network?? weight?? ???? #     ?? ??  ??!
        self.gdqn_target.load_state_dict(weights)  # target model network? weight?? ??? ???? ??


    def set_agent_info(self, agent, pos, view_range):

        if agent[9] == "1": #1?? predator ??
            self.idx = int(agent[11:])
            #print("self.idx predator1??############################################",self.idx)
            self.adj = torch.ones(predator1_adj)

            self.pos = pos
            self.view_range = view_range

        else:               #2?? predator ??
            self.idx = int(agent[11:]) + n_predator1
            #print("self.idx predator2??############################################", self.idx)
            self.adj = torch.ones(predator2_adj)

            self.pos = pos
            self.view_range = view_range


        self.gdqn= self.gdqns[self.idx]
        self.gdqn_target = self.gdqn_targets[self.idx]
        self.gdqn_optimizer = self.gdqn_optimizers[self.idx]
        self.buffer = self.buffers[self.idx]


    def set_agent_buffer(self ,idx):
        self.buffer = self.buffers[idx]

    def from_guestbook(self):
        x_start = self.pos[0] + self.view_range

        y_start = self.pos[1] + self.view_range



        x_range = int(self.view_range)
        y_range = int(self.view_range)
        z_range = self.entire_state[2]


        extracted_area = self.shared[x_start - x_range:x_start + x_range, y_start - y_range: y_start + y_range,
                         :z_range]



        return extracted_area

    # def from_guestbook(self):  # 에이전트의 pos 정보를 받아서 정보를 가져오는 함수 pos:에이전트의 절대 위치 pos: 리스트 shared: 방명록
    #     x_start = self.pos[0] + 10
    #
    #     y_start = self.pos[1] + 10
    #
    #     z_start = 0
    #
    #     x_range = int(self.view_range)  # 사실 view_range=5라고 했을 때, 10107의 obs를 얻는데, agent의 좌표가 정중앙인가...?에 하는 의심 일단 믿어.ㅠㅠ
    #     y_range = int(self.view_range)
    #     z_range = self.entire_state[2]  # feature_dim 을 가져오는 것
    #
    #     extracted_area = self.shared[x_start - x_range:x_start + x_range, y_start - y_range: y_start + y_range,
    #                      :z_range]
    #
    #     # 구석에 있는 agent들이 observation을 어떻게 가지고 올지 확인하고 수정해야 할 필요 았음
    #
    #     return extracted_area  # (887)으로 출력

    def to_guestbook(self, info):

        x_start = self.pos[0] + self.view_range
        y_start = self.pos[1] + self.view_range


        x_range = int(self.view_range)  # ?? view_range=5?? ?? ?, 10107? obs? ???, agent? ??? ?????...?? ?? ?? ?? ??.??
        y_range = int(self.view_range)


        self.shared[x_start - x_range:x_start + x_range, y_start - y_range: y_start + y_range, :] += info


        self.shared[0:self.view_range, :, :] = 0 #윗줄
        self.shared[self.entire_state[0]-self.view_range:self.entire_state[0], :, :] = 0 #아랫줄
        self.shared[:, 0:self.view_range, :] = 0 #왼쪽줄
        self.shared[:, self.entire_state[0]-self.view_range:self.entire_state[0], :] = 0 #오른쪽줄

    def shared_decay(self):
        self.shared = self.shared * 0.97


    def get_action(self, state, mask=None):
        book = self.from_guestbook() #self.pos? ??? ?? ???? shared graph?? ??? ????.
        q_value, shared_info = self.gdqn(torch.tensor(state).to(self.device), self.adj.to(self.device), book.to(self.device)) #shared_info : shared graph? ????? ? ???

        self.to_guestbook(shared_info.to('cpu')) #shared_graph? ??? ??? ????.
        self.epsilon *= args.eps_decay  # ??? args? eps_decay? ??? ?? ????? ?
        self.epsilon = max(self.epsilon, args.eps_min)  # ??? args_eps ?? ????? ??? ???? ??? ??
        # predict?? state? ?? ? action? ??? ????, ? ??? ???? ?
        if np.random.random() < self.epsilon:  # 0?? 1?? ??? ??? ????, ? ?? ????? ??? action? ???? ???.
            return random.randint(0, self.dim_act - 1), book  # ??? 0?? action_dim-1 ??? ????? ???? ?? ?? ??.
        return torch.argmax(q_value).item() , book
        #return np.argmax(q_value)  # ?? ?? ????, q_value ? ?? ?? ?? ???? ?? ????.

        try:
            torch.cuda.empty_cache()
        except:
            pass

    def replay(self): #????? ??? ? ? ???....??? unsqueeze ????
        for _ in range(10):
            self.gdqn_optimizer.zero_grad()

            observations, book, actions, rewards, next_observations, book_next, termination, truncation = self.buffer.sample()  # ?? ??? buffer?? ??? sample? ??

            next_observations = torch.tensor(next_observations)
            observations = torch.tensor(observations)

            next_observations = next_observations.reshape(-1,self.entire_state[2])
            observations = observations.reshape(-1,self.entire_state[2])

            # to device
            observations = observations.to(self.device)
            next_observations = next_observations.to(self.device)
            adj = self.adj.to(self.device)

            q_values, _ = self.gdqn(observations.unsqueeze(0), adj.unsqueeze(0), book[0].detach().to(self.device))# ??? observation??? q ?
            q_values = q_values[0][actions]
            #? adj ? unsqueeze ? ????? ??. ??? 36*36 ? 1*36*36 ?? ???? densesageconv? ? ? ?? ? ??.

            next_q_values, _ = self.gdqn_target(next_observations.unsqueeze(0), adj.unsqueeze(0), book_next[0].detach().to(self.device))  # next state? ??? target ?
            next_q_values = torch.max(next_q_values)  # next state? ??? target ?

            targets = int(rewards[0]) + (1 - int(termination[0])) * next_q_values * args.gamma
            loss = self.criterion(q_values, targets.detach())
            #loss.backward(retain_graph=True)
            loss.backward()
            #loss.backward()
            self.gdqn_optimizer.step()


            try:
                torch.cuda.empty_cache()
            except:
                pass
    def reset_shred(self,shared):
        self.shared = shared
