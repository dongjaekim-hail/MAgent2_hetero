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




tf.keras.backend.set_floatx('float64')
# wandb.init(name='DQN', project="deep-rl-tf2")

parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.95)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--eps', type=float, default=1.0)
parser.add_argument('--eps_decay', type=float, default=0.995)
parser.add_argument('--eps_min', type=float, default=0.01)

args = parser.parse_args()  #Namespace(gamma=0.95, lr=0.005, batch_size=32, eps=1.0, eps_decay=0.995, eps_min=0.01)


entire_state = (45,45,7)
predator1_obs = (8,8,7)
predator2_obs = (10,10,7)
dim_act = 13
n_predator1 = 10
n_predator2 = 10
eps_decay = 0.1
batch_size = 10


# 문자열을 역으로 순회하며 첫 번째 나타나는 '_'을 찾음. test 파일 돌려보면 agent 가 "predator_0_23" 꼴로 나오는데, 해당 에이전트의 gnn 및 dqn을 지정해야해서 23을 뽑도록..
def get_agent_idx(agent): #predator_2_4

    if agent[9] == "1":
        return int(agent[11:])
    else: #집단 2 일때
        return int(agent[11:])+n_predator1 #predator2는 predator1 바로 뒤에 append 되어있기 때문에 n_predator1 을 더해주는 것!



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

    def get_action(self, state, adj, mask = None, from_guestbook, gdqns):
        q_value, shared = gdqns(self, state, adj, mask=None, from_guestbook)[0]
        self.epsilon *= args.eps_decay  # 기존의 args에 eps_decay를 곱해서 다시 저장하라는 말
        self.epsilon = max(self.epsilon, args.eps_min)  # 그리고 args_eps 값의 최소값으로 정해진 것보다는 크도록 설정
          # predict에는 state에 따른 각 action의 값들이 나올건데, 그 값들을 저장하는 것
        if np.random.random() < self.epsilon:  # 0부터 1까지 랜덤한 숫자를 생성하고, 그 수가 입실론보다 작으면 action을 랜덤으로 뽑는다.
            return random.randint(0, self.dim_act - 1)  # 그래서 0부터 action_dim-1 까지의 정수중에서 랜덤으로 하나 뽑는 거다.
        return np.argmax(q_value)  # 만약 그게 아니라면, q_value즉 가장 크게 하는 인덱스의 값을 추출한다.


    def replay(self,buffer, gdqn_target, gdqn_optimizer):
        for _ in range(10):
            states, actions, rewards, next_states, done = self.buffer.sample()  # 위의 생성한 buffer에서 하나의 sample을 뽑음
            targets = gdqn_target(state, adj, mask = None, from_guestbook)  # target network으로부터 target을 만들어야 하므로
            next_q_values = gdqn_target(next_states,adj, mask = None, from_guestbook).max(axis=1)  # next state에 대해서도 q value을 예측
            targets[range(args.batch_size), actions] = rewards + (1 - done) * next_q_values * args.gamma
            loss = self.criterion(states, targets)
            loss.backward
            gdqn_optimizer.step()


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




