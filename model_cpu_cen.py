import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DenseSAGEConv
import numpy as np
from collections import deque
import torch.optim as optim
from arguments import args
import random
import arguments

class G_DQN(nn.Module):
    def __init__(self,  dim_act, observation_state):
        super(G_DQN, self).__init__()
        self.eps_decay = args.eps_decay

        self.observation_state = observation_state #전채 state (25*25*3)
        print(self.observation_state)
        self.dim_act = dim_act

        #GRAPH
        self.dim_feature = self.observation_state[2] #3
        self.gnn1 = DenseSAGEConv(self.dim_feature, 128)
        self.gnn2 = DenseSAGEConv(128, self.dim_feature)
        #self.sig = nn.Sigmoid() #sigmoid 는 아마 필요 없을 듯!

        #DQN
        self.dim_input = self.observation_state[0] * self.observation_state[1] * self.observation_state[2]
        print(self.dim_input)
        self.FC1 = nn.Linear(self.dim_input, 128)
        self.FC2 = nn.Linear(128, dim_act)
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.ReLU()


    def forward(self, x, adj): #info 는 필요 없음

        try:
            torch.cuda.empty_cache()
        except:
            pass


        if isinstance(x, np.ndarray):
            x = torch.tensor(x).float()
        else:
            pass

        x = self.gnn1(x.reshape(-1, self.dim_feature), adj)
        x = F.elu(self.gnn2(x,adj)).squeeze()  # exponential linear unit #squeeze 를 하는 이유: x가 batch_size를 고려해서 받을 수 있도록 설계 됐기 때문에 1*100*3꼴로 나옴->100*3으로 바꿔주기 위함

        #x = F.elu(self.gnn2(self.gnn1(x.reshape(-1, self.dim_feature) ,adj),adj)).squeeze()
        #dqn = x[:, :self.dim_feature]  #   100*3 : 위의 x중 절반은 dqn 으로 들어가고 나머지 절반은 sigmoid취해서 가져갈 것만 기록하도록 한다.
        # shared = self.sig(x[:, self.dim_feature:]) #share graph 로 들어갈 것! 가져오고
        # shared = dqn * shared # sigmoid 해준 값과 x를 dot곱해줌
        # shared = shared.reshape(self.observation_state).detach() #다시 10*10*5 꼴로 만들어주어야 함-> 이를 shared graph 에 넘겨주어야 한다.


        x = x.reshape(-1, self.dim_input) #쭉 펴주고
        print(x.shape,"#"*100)
        x = self.FC1(x)
        x = self.FC2(x)

        return x




class ReplayBuffer:
   def __init__(self, capacity=10000):
      self.buffer = deque(maxlen=capacity)

   def put(self, observation, action , reward, next_observation, termination, truncation):
       self.buffer.append([observation, action , reward, next_observation, termination, truncation]) #[state, action, reward, next_state, done]리스트 형태로 history를 저장



   def sample(self):
       sample = random.sample(self.buffer, 1)  # batch size만큼 buffer에서 가져온다.

       observation, book , action , reward, next_observation, book_next, termination, truncation = zip(*sample)
       return observation, book , action , reward, next_observation, book_next, termination, truncation  # buffer에서 데이터 받아서 반환하는 과정을 거침

   def size(self):
      return len(self.buffer)   #buffer 사이즈길이만큼 뱉는 것


class DQN(nn.Module):
    def __init__(self, dim_act, observation_state):
        super(DQN, self).__init__()
        self.observation_state = observation_state
        self.dim_act = dim_act

        # DQN
        self.dim_input = np.prod(observation_state)
        self.FC1 = nn.Linear(self.dim_input, 128)
        self.FC2 = nn.Linear(128, dim_act)
        self.relu = nn.ReLU()

    def forward(self, state):
        if isinstance(state, np.ndarray):
            state = torch.tensor(state).float()
        else:
            pass

        x = state.reshape(-1, self.dim_input)
        x = self.relu(self.FC1(x))
        x = self.FC2(x)

        return x


class ReplayBuffer_cen:
   def __init__(self, capacity=10000):
      self.buffer = deque(maxlen=capacity)

   def put(self, observation , action , reward, next_observation, termination, truncation):
       self.buffer.append([observation, action , reward, next_observation, termination, truncation]) #[state, action, reward, next_state, done]리스트 형태로 history를 저장

#

   def sample(self):
       sample = random.sample(self.buffer, 1)  # batch size만큼 buffer에서 가져온다.

       observation , action , reward, next_observation, termination, truncation = zip(*sample)
       return observation , action , reward, next_observation, termination, truncation  # buffer에서 데이터 받아서 반환하는 과정을 거침

   def size(self):
      return len(self.buffer)   #buffer 사이즈길이만큼 뱉는 것