# these codes modified from https://github.com/marload/DeepRL-TensorFlow2
# import wandb # this one, you need to do some stuff. wandb login.
import gymnasium as gym
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten
from tensorflow.keras.optimizers import Adam

import gym
import argparse
import numpy as np
from collections import deque
import random

import dgl
from dgl.nn import DenseSAGEConv

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


class ReplayBuffer:                 #슈도코드를 보면 알겠지만, 애초에 history를 저장해서 그로부터 하나씩 가져와서 학습을 진행시킨다.
   def __init__(self, capacity=10000):
      self.buffer = deque(maxlen=capacity)    #deque의 알고리즘을 가지는 객체 생성
                                    #append를 통해 가장 오른쪽에 데이터를 추가하고 appendleft를 통해 왼쪽에 추가한다.
                                    #maxlen을 넘으면 자동으로 왼쪽에서 삭제된다.

   def put(self, state, action, reward, next_state, done):
      self.buffer.append([state, action, reward, next_state, done]) #[state, action, reward, next_state, done]리스트 형태로 history를 저장

   def sample(self):
      sample = random.sample(self.buffer, args.batch_size)   #batch size만큼 buffer에서 가져온다.
      states, actions, rewards, next_states, done = map(np.asarray, zip(*sample))
      states = np.array(states).reshape(args.batch_size, -1)
      next_states = np.array(next_states).reshape(args.batch_size, -1)
      return states, actions, rewards, next_states, done     #buffer에서 데이터 받아서 반환하는 과정을 거침

   def size(self):
      return len(self.buffer)   #buffer 사이즈길이만큼 뱉는 것(?)

class SharedGnn:
   def  __init__(self,input_dim,out_dim):
      self.shared=DenseSAGEConv(input_dim,out_dim)



class ActionStateModel:
   def __init__(self, state_dim, aciton_dim):
      self.state_dim = state_dim
      self.action_dim = aciton_dim
      self.epsilon = args.eps     #args라는 전역변수같은 녀석에 eps가 들어있나봄->맞음

      self.model = self.create_model()

   def create_model(self):   #input에는 state_dim  output에는 action_dim 뱉음 중간 충은 노드100개, 각 노드에 렐루닮
      model = tf.keras.Sequential([
         Input((self.state_dim,)),
         Dense(100, activation='relu'),
         Dense(self.action_dim)
      ])
      model.compile(loss='mse', optimizer=Adam(args.lr))  #loss를 mse로 하는 이유는 목적함수가 mse의 형태이기 때문이다.
      return model

   def predict(self, state):   #state라는 인자를 넘겨주면, 앞에서 정의한 model에 predict 함수를 정의해서 결과를 뽑아내는 함수이다.
      return self.model.predict(state)

   def get_action(self, state):
      state = np.reshape(state, [1, self.state_dim])
      self.epsilon *= args.eps_decay
      self.epsilon = max(self.epsilon, args.eps_min)
      q_value = self.predict(state)[0]
      if np.random.random() < self.epsilon:
         return random.randint(0, self.action_dim - 1)
      return np.argmax(q_value)

   def train(self, states, targets):
      self.model.fit(states, targets, epochs=1, verbose=0)



class Agent:
   def __init__(self, env):
      self.env = env
      self.action_dim = act_dim 넣고
      self.
      #gnn attention weight-> observation을 받고, 그 observation state 들의 값을 각각 얼만큼의 비중으로 가져올 것인지 (유사 attention)
      #gaw 에는 에이전트 본인만의 weight만 넣는다.
      self.gaw=
      #각 에이전트들의 gnn네트워크 -> 주변 에이전트들로부터 정보를 취합하는 과정을 나타낸다.
      self.state_dim = np.prod(gnn거쳐서 나온 출력값)


      self.model = ActionStateModel(self.state_dim, self.action_dim)
      self.target_model = ActionStateModel(self.state_dim, self.action_dim)



      self.buffer = ReplayBuffer()   #ReplayBuffer객체 형성

   def target_update(self):
      weights = self.model.model.get_weights()       #behavior network에서 weight들을 가져오고
      self.target_model.model.set_weights(weights)   #target model network의 weight들에 그대로 복사하는 과정

   def replay(self):
      for _ in range(10):
         states, actions, rewards, next_states, done = self.buffer.sample()  #위의 생성한 buffer에서 하나의 sample을 뽑음
         targets = self.target_model.predict(states)   #target network으로부터 target을 만들어야 하므로
         next_q_values = self.target_model.predict(next_states).max(axis=1) #next state에 대해서도 q value을 예측
         targets[range(args.batch_size), actions] = rewards + (1 - done) * next_q_values * args.gamma
         self.model.train(states, targets)  #그렇게 targets 에 대한 y value을 구해서 그에 대해서 train을 진행시킴

   def train(self, max_episodes=1000, render_episodes=100):
      for ep in range(max_episodes):
         done, total_reward = False, 0
         state, _ = self.env.reset()
         while not done:   #한 에피소드에서 매 step 별로 진행한다는 말
            action = self.model.get_action(state)
            next_state, reward, done, _, _ = self.env.step(action)
            self.buffer.put(state, action, reward * 0.01, next_state, done)
            total_reward += reward
            state = next_state							#에피소드 끝날때까지 버퍼에 경험들을 모으는 과정
         if self.buffer.size() >= args.batch_size:		#다 모으고 나서...
            self.replay()
         self.target_update()
         print('EP{} EpisodeReward={}'.format(ep, total_reward))
         # wandb.log({'Reward': total_reward})
         if (ep+1) % render_episodes == 0:
            state, _ = self.env.reset()
            while not done:
               self.env.render()
               action = self.model.get_action(state)
               next_state, reward, done, _, _  = self.env.step(action)



def main():
   env = gym.make("Pong-v4", render_mode='human')
   agent = Agent(env)                          #Agent의 init: __init__(self, env) ->env를 받아야함
   agent.train(max_episodes=1000, render_episodes = 100)


if __name__ == "__main__":
   main()

