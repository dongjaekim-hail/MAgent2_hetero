from model import G_DQN, ReplayBuffer
import numpy as np
from copy import deepcopy
import random
import torch
from torch.optim import Adam
import argparse
from collections import deque
import gymnasium as gym
import tensorflow as tf
import argparse
from arguments import args



entire_state = (45,45,7)
predator1_obs = (8,8,7)
predator2_obs = (10,10,7)
dim_act = 13
n_predator1 = 10
n_predator2 = 10
eps_decay = 0.1
batch_size = 10



class MADQN():  # def __init__(self,  dim_act, observation_state):
    def __init__(self, n_predator1, n_predator2, predator1_obs, predator2_obs ,dim_act, entire_state):

        self.predator1_obs = predator1_obs
        self.predator2_obs = predator2_obs
        self.n_predator1 = n_predator1
        self.n_predator2 = n_predator2
        self.dim_act= dim_act

        # 초반 n_predator1 개는 predator1의 dqn 이고, 그 뒤의 것은 predator2 의 dqn 둘의 observation 이 다르기 때문에 다른 dqn을 사용해야 한다.
        self.gdqns = [G_DQN(self.dim_act, self.predator1_obs) for _ in range(self.n_predator1)] +[G_DQN( self.dim_act,self.predator2_obs) for _ in range(self.n_predator2)]
        self.gdqns_target = [G_DQN(self.dim_act, self.predator1_obs) for _ in range(self.n_predator1)] +[G_DQN( self.dim_act,self.predator2_obs) for _ in range(self.n_predator2)] #학습의 안정을 위해 target dqn 설정
        self.buffer = [ReplayBuffer() for _ in range(self.n_predator1+self.n_predator2)]
        self.gdqns_optimizer = [Adam(x.parameters(), lr=0.001) for x in self.gdqns]
        self.target_optimizer = [Adam(x.parameters(), lr=0.001) for x in self.gdqns_target]

        #shared gnn 의 부분!
        self.shared = np.zeros(self.entire_state)



    def target_update(self): #주기적으로 target 업데이트 함 tensorflow
        for i in range(n_predator1 + n_predator2):
            weights = self.gdqns[i].get_weights()  # behavior network에서 weight들을 가져오고
            self.gdqns_target[i].set_weights(weights)  # target model network의 weight들에 그대로 복사하는 과정


    def from_guestbook(self, shared, pos, range, entire_state): #에이전트의 pos 정보를 받아서 정보를 가져오는 함수 pos:에이전트의 절대 위치 pos: 리스트 shared: 방명록
        x_start = pos[0]
        y_start = pos[1]
        z_start = 0

        x_size = range
        y_size = range
        z_size = entire_state[2]

        extracted_area = shared[x_start:x_start + x_size, y_start:y_start + y_size, z_start : z_start + z_size] #pos의 값을 정 중앙의 값이므로 수정할 필요 있음
        return extraced_area #(8*8*7)으로 출력

    def to_guestbook(self,  shared, pos, range, entire_state, info):
        # 에이전트의 Pos 정보를 받아서 shared graph에 정보를 저장하는 함수, info: forward를 거쳐서 나온 기록할 정보, pos: 에이전트의 절대 위치, shared :방명록
        x_start = pos[0]
        y_start = pos[1]
        z_start = 0

        x_size = range
        y_size = range
        z_size = entire_state[2]

        shared[x_start-x_size:x_start + x_size, y_start-y_size:y_start + y_size, z_start:z_start + z_size] += info

    def get_action(self, state, adj, mask = None, extraced_area, idx):
        book = from_guestbook(self.shared,pos)
        q_value, shared = self.gdqns[idx](self, state, adj, mask=None, from_guestbook)
        self.epsilon *= args.eps_decay  # 기존의 args에 eps_decay를 곱해서 다시 저장하라는 말
        self.epsilon = max(self.epsilon, args.eps_min)  # 그리고 args_eps 값의 최소값으로 정해진 것보다는 크도록 설정
          # predict에는 state에 따른 각 action의 값들이 나올건데, 그 값들을 저장하는 것
        if np.random.random() < self.epsilon:  # 0부터 1까지 랜덤한 숫자를 생성하고, 그 수가 입실론보다 작으면 action을 랜덤으로 뽑는다.
            return random.randint(0, self.dim_act - 1)  # 그래서 0부터 action_dim-1 까지의 정수중에서 랜덤으로 하나 뽑는 거다.
        return np.argmax(q_value)  # 만약 그게 아니라면, q_value 증 가장 크게 하는 인덱스의 값을 추출한다.


    def replay(self, gdqn_optimizer, idx, book):
        for _ in range(10):
            states, actions, rewards, next_states, done = self.buffer[idx].sample()  # 위의 생성한 buffer에서 하나의 sample을 뽑음
            targets, _ = self.gdqn_target[idx](state, adj, mask = None, book)  # target network으로부터 target을 만들어야 하므로
            next_q_values, _ = self.gdqns_target[idx](next_states, adj, mask = None, from_guestbook).max(axis=1)  # next state에 대해서도 q value을 예측
            targets[range(args.batch_size), actions] = rewards + (1 - done) * next_q_values * args.gamma
            loss = self.criterion(states, targets)
            loss.backward
            self.gdqn_optimizer[idx].step()


    def train(self, max_episodes=1000, render_episodes=100,gdqns): # max_episodes=1000, render_episodes=100 수정필요  env 를 어떻게 받아올지도 생각해야함
        for ep in range(max_episodes):
            done, total_reward = False, 0
            state, _ = self.env.reset() #수정필요
            while not done:  # 한 에피소드에서 매 step 별로 진행한다는 말
                action = get_action(self, gdqns, state, adj, mask = None, from_guestbook)
                next_state, reward, done, _, _ = self.env.step(action) #수정필요
                self.buffer.put(state, action, reward * 0.01, next_state, done)
                total_reward += reward
                state = next_state  # 에피소드 끝날때까지 버퍼에 경험들을 모으는 과정
            if self.buffer.size() >= args.batch_size:  # 다 모으고 나서...
                self.replay() #수정필요 내용 :
            self.target_update() #수정필요
            print('EP{} EpisodeReward={}'.format(ep, total_reward))
            # wandb.log({'Reward': total_reward})
            if (ep + 1) % render_episodes == 0:
                state, _ = self.env.reset()
                while not done:
                    self.env.render()
                    action = gdqns.get_action(state)
                    next_state, reward, done, _, _ = self.env.step(action)




