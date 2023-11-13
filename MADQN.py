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
predator1_adj = (100,100)
predator2_adj = (36,36)
#agent = None



class MADQN():  # def __init__(self,  dim_act, observation_state):
    def __init__(self, n_predator1, n_predator2, predator1_obs, predator2_obs, dim_act , entire_state, shared):
        self.entire_state = entire_state
        self.predator1_obs = predator1_obs
        self.predator2_obs = predator2_obs
        self.n_predator1 = n_predator1
        self.n_predator2 = n_predator2
        self.dim_act = dim_act
        self.epsilon = args.eps
        self.eps_decay = args.eps_decay

        # 초반 n_predator1 개는 predator1의 dqn 이고, 그 뒤의 것은 predator2 의 dqn 둘의 observation 이 다르기 때문에 다른 dqn을 사용해야 한다.
        self.gdqns = [G_DQN(self.dim_act, self.predator1_obs) for _ in range(self.n_predator1)] + [
            G_DQN(self.dim_act, self.predator2_obs) for _ in range(self.n_predator2)]
        self.gdqn_targets = [G_DQN(self.dim_act, self.predator1_obs) for _ in range(self.n_predator1)] + [
            G_DQN(self.dim_act, self.predator2_obs) for _ in range(self.n_predator2)]  # 학습의 안정을 위해 target dqn 설정
        self.buffers = [ReplayBuffer() for _ in range(self.n_predator1 + self.n_predator2)]
        self.gdqn_optimizers = [Adam(x.parameters(), lr=0.001) for x in self.gdqns]
        #self.target_optimizer = [Adam(x.parameters(), lr=0.001) for x in self.gdqns_target] #이게 필요하지는 않지 어차피 weight받아와서 업데이트 하는건데
        self.criterion = nn.MSELoss()

        # shared gnn 의 부분!
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




    def target_update(self):  # 주기적으로 target 업데이트 함 tensorflow 수정해야함
        weights = self.gdqn.state_dict()  # behavior network에서 weight들을 가져오고 #     없는 함수  변경!
        self.gdqn_target.load_state_dict(weights)  # target model network의 weight들에 그대로 복사하는 과정


    def set_agent_info(self, agent, pos, view_range):

        if agent[9] == "1": #1번째 predator 집단
            self.idx = int(agent[11:])
            #print("self.idx predator1확인############################################",self.idx)
            self.adj = torch.ones(predator1_adj)

            self.pos = pos
            self.view_range = view_range

        else:               #2번째 predator 집단
            self.idx = int(agent[11:]) + n_predator1
            #print("self.idx predator2확인############################################", self.idx)
            self.adj = torch.ones(predator2_adj)

            self.pos = pos
            self.view_range = view_range


        self.gdqn= self.gdqns[self.idx]
        self.gdqn_target = self.gdqn_targets[self.idx]
        self.gdqn_optimizer = self.gdqn_optimizers[self.idx]
        self.buffer = self.buffers[self.idx]

    def from_guestbook(self):  # 에이전트의 pos 정보를 받아서 정보를 가져오는 함수 pos:에이전트의 절대 위치 pos: 리스트 shared: 방명록
        x_start = self.pos[0] + 10

        y_start = self.pos[1] + 10

        z_start = 0

        x_range = int(self.view_range)  # 사실 view_range=5라고 했을 때, 10107의 obs를 얻는데, agent의 좌표가 정중앙인가...?에 하는 의심 일단 믿어.ㅠㅠ
        y_range = int(self.view_range)
        z_range = self.entire_state[2]  # feature_dim 을 가져오는 것

        extracted_area = self.shared[x_start - x_range:x_start + x_range, y_start - y_range: y_start + y_range,
                         :z_range]

        # 구석에 있는 agent들이 observation을 어떻게 가지고 올지 확인하고 수정해야 할 필요 았음

        return extracted_area  # (887)으로 출력


    def to_guestbook(self, info):  # info : gnn을 거쳐서 sigmoid취해준 결과를 곱해준 값
        # 에이전트의 Pos 정보를 받아서 shared graph에 정보를 저장하는 함수, info: forward를 거쳐서 나온 기록할 정보, pos: 에이전트의 절대 위치, shared :방명록
        # shared 와 info 모두 3차원형태

        x_start = self.pos[0] + 10
        y_start = self.pos[1] + 10
        z_start = 0

        x_range = int(self.view_range)  # 사실 view_range=5라고 했을 때, 10107의 obs를 얻는데, agent의 좌표가 정중앙인가...?에 하는 의심 일단 믿어.ㅠㅠ
        y_range = int(self.view_range)
        z_range = self.entire_state[2]  # feature_dim 을 가져오는 것

        self.shared[x_start - x_range:x_start + x_range, y_start - y_range: y_start + y_range, :z_range] += info

        # shared 배열에서 해당 부분을 0으로 설정합니다.
        self.shared[0:10, 0:10, :z_range] = 0
        self.shared[55:65, 55:65, :z_range] = 0


    def get_action(self, state, mask=None):
        book = self.from_guestbook() #self.pos에 기록된 값을 참고하여 shared graph에서 정보를 가져오는데,,,여기서 문제가 발생한다.
        q_value, shared_info = self.gdqn(state, self.adj, book) #shared_info : shared graph에 넘겨주어야 할 정보들

        self.to_guestbook(shared_info) #shared_graph에 받아온 정보를 넘겨준다.
        self.epsilon *= args.eps_decay  # 기존의 args에 eps_decay를 곱해서 다시 저장하라는 말
        self.epsilon = max(self.epsilon, args.eps_min)  # 그리고 args_eps 값의 최소값으로 정해진 것보다는 크도록 설정
        # predict에는 state에 따른 각 action의 값들이 나올건데, 그 값들을 저장하는 것
        if np.random.random() < self.epsilon:  # 0부터 1까지 랜덤한 숫자를 생성하고, 그 수가 입실론보다 작으면 action을 랜덤으로 뽑는다.
            return random.randint(0, self.dim_act - 1)  # 그래서 0부터 action_dim-1 까지의 정수중에서 랜덤으로 하나 뽑는 거다.
        return torch.argmax(q_value).item()
        #return np.argmax(q_value)  # 만약 그게 아니라면, q_value 증 가장 크게 하는 인덱스의 값을 추출한다.

    def replay(self): #배치사이즈 살려야 할 것 같은데....나중에 unsqueeze 없애야함
        for _ in range(10):
            book = self.from_guestbook()
            observations, actions, rewards, next_observations, termination, truncation = self.buffer.sample()  # 위의 생성한 buffer에서 하나의 sample을 뽑음

            next_observations = torch.tensor(next_observations)
            observations = torch.tensor(observations)

            next_observations = next_observations.reshape(-1,3)
            observations = observations.reshape(-1,3)

            self.gdqn_optimizer.zero_grad()

            q_values, _ = self.gdqn(observations.unsqueeze(0), (self.adj).unsqueeze(0), book.detach())# 실제로 observation에서의 q 값
            q_values = q_values[0][actions]
#아 adj 를 unsqueeze 를 해야하는것 같다. 그래서 36*36 을 1*36*36 으로 해주어야 densesageconv를 할 수 있는 것 같다.

            next_q_values, _ = self.gdqn_target(next_observations.unsqueeze(0), (self.adj).unsqueeze(0), book)  # next state에 대해서 target 값
            next_q_values = torch.max(next_q_values)  # next state에 대해서 target 값

            targets = int(rewards[0]) + (1 - int(termination[0])) * next_q_values * args.gamma
            loss = self.criterion(q_values, targets.detach())
            #loss.backward(retain_graph=True)
            loss.backward()
            #loss.backward()
            self.gdqn_optimizer.step()


    def reset_shred(self,shared):
        self.shared = shared



