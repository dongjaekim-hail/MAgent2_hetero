import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DenseSAGEConv
import numpy as np
from collections import deque
import torch.optim as optim
from arguments import args
import random

class G_DQN(nn.Module):
    def __init__(self,  dim_act, observation_state):
        super(G_DQN, self).__init__()
        self.eps_decay = args.eps_decay

        self.observation_state = observation_state
        self.dim_act = dim_act

        #GRAPH
        self.dim_feature = observation_state[2]
        self.gnn1 = DenseSAGEConv(self.dim_feature, 128)
        self.gnn2 = DenseSAGEConv(128, self.dim_feature*2) #feature 의 2배를 출력
        self.sig = nn.Sigmoid() #sigmoid 도 이렇게 해야 하고..

        #DQN
        self.dim_input = observation_state[0] * observation_state[1] * observation_state[2]*2 #concat 해서 2배!
        self.FC1 = nn.Linear(self.dim_input, 128)
        self.FC2 = nn.Linear(128, dim_act)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, state, adj, info): #x외 adj는 밖에서 넣어줘야 되고  GSAGE에 입력값 넣어주면 출력값 뱉고, from_guestbook 아예 크기에 맞는 (8*8*7)의 형태로 넣어주고

        if isinstance(state, np.ndarray):
            state = torch.tensor(state).float()
        else:
            pass

        print("state뭔데?",state.shape)
        #state = state.squeeze()

        x = state.reshape(-1, self.dim_feature) #(10*10*3)를 (100*3)으로 변경하여 그래프의 featur metrix으로 바꾸어주는 역활!
        x = self.gnn1(x,adj) #노드끼리 fully connective 되어 있다는 가정아래!gnn어차피 fully connective 여서 mask 인자 없애버림
        x = self.gnn2(x,adj)
        x = F.elu(x)  # exponential linear unit
        x = x.squeeze() #squeeze 를 하는 이유: x가 batch_size를 고려해서 받을 수 있도록 설계 됐기 때문에 1*100*14꼴로 나옴->100*14으로 바꿔주기 위함

        dqn = x[:, :self.dim_feature]  #   100*7 : 위의 x중 절반은 dqn 으로 들어가고 나머지 절반은 sigmoid취해서 가져갈 것만 기록하도록 한다.
        print("dqn",dqn.shape)

        shared = self.sig(x[:, self.dim_feature:]) #share graph 로 들어갈 것! 가져오고
        shared = dqn * shared # sigmoid 해준 값과 x를 dot곱해줌
        shared = shared.reshape(self.observation_state) #다시 10*10*5 꼴로 만들어주어야 함-> 이를 shared graph 에 넘겨주어야 한다.


        input = torch.cat((shared, info), dim=0) #현재 info가 5*5*7 이라서 concatenate이 안됨->수정함
        print("info", info.shape)
        print("shared", shared.shape)
        print("input",input.shape)


        x = input.reshape(-1, self.dim_input) #쭉 펴주고
        x = self.FC1(x)
        x = self.FC2(x)

        return x, shared #shared_graph에 넣는건 밖에서 진행하자.




class ReplayBuffer:                 #슈도코드를 보면 알겠지만, 애초에 history를 저장해서 그로부터 하나씩 가져와서 학습을 진행시킨다. 수정필요!
   def __init__(self, capacity=10000):
      self.buffer = deque(maxlen=capacity)    #deque의 알고리즘을 가지는 객체 생성
                                    #append를 통해 가장 오른쪽에 데이터를 추가하고 appendleft를 통해 왼쪽에 추가한다.
                                    #maxlen을 넘으면 자동으로 왼쪽에서 삭제된다.

   def put(self, observation, action, reward, next_observation, termination, truncation):
      self.buffer.append([observation, action, reward, next_observation, termination, truncation]) #[state, action, reward, next_state, done]리스트 형태로 history를 저장



   def sample(self):
       sample = random.sample(self.buffer, 1)  # batch size만큼 buffer에서 가져온다.

       observation, action, reward, next_observation, termination, truncation = zip(*sample)
       return observation, action, reward, next_observation, termination, truncation  # buffer에서 데이터 받아서 반환하는 과정을 거침

   def size(self):
      return len(self.buffer)   #buffer 사이즈길이만큼 뱉는 것
