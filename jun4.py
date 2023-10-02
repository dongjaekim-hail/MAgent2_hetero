# def get_agent_idx(string):
#     # 문자열을 역으로 순회하며 첫 번째 나타나는 '_'을 찾습니다.
#     for char in reversed(string):
#         if char == '_':
#             # '_'를 찾으면 이후의 문자열을 추출하고 숫자로 변환합니다.
#             last_part = string[string.rindex('_') + 1:]
#             return int(last_part)
#
#
#
#
# # 예제 사용
# input_string = "predator_1_68"
# last_part_as_int = get_agent_idx(input_string)
# print(last_part_as_int)  # 출력: 18
#

# def get_agent_idx(agent):
#     # 문자열을 역으로 순회하며 첫 번째 나타나는 '_'을 찾음.
#     for char in reversed(agent):
#         if char == '_':
#             # '_'를 찾으면 이후의 문자열을 추출하고 숫자로 변환.
#             last_part = agent[agent.rindex('_') + 1:]
#             return int(last_part)
#
# # 예제 사용
# input_string = "predator_1_68"
# last_part_as_int = get_agent_idx(input_string)
# print(last_part_as_int)  # 출력: 18
#



# feat = th.ones(6, 10)
# adj = th.tensor([[0., 0., 1., 0., 0., 0.],
#         [1., 0., 0., 0., 0., 0.],
#         [0., 1., 0., 0., 0., 0.],
#         [0., 0., 1., 0., 0., 1.],
#         [0., 0., 0., 1., 0., 0.],
#         [0., 0., 0., 0., 0., 0.]])
import torch as th
# import torch
#
# # # 예를 들어, 0부터 9까지의 정수값을 가지는 2x3x4 크기의 텐서 생성
# low = 0
# high = 10
# size = (2, 3, 4)
# a = torch.randint(low, high, size)
# print(a)
# a = a.view(-1,4)
# print(a)
#
# a = a.view(2,3,4)
# print(a)
#다시 원상복구 되느데
#
# # low와 high 사이의 정수값이 생성됩니다.
#
# # def sum
# n_predator1=10
# def get_agent_idx(agent):  # predator_2_4
#
#     if agent[9] == "1":
#         return int(agent[11:])
#     else:
#         return int(agent[11:]) + n_predator1
#
# print(get_agent_idx("predator_2_4"))
#
#
import numpy as np
#
#45x45x7 크기의 3차원 배열을 가정합니다.
#이 배열은 예시이므로 실제 데이터를 사용하십시오.
# np.random.seed(1)
# state = np.random.randint(0, 50, size=(3, 3, 1), dtype=np.uint8)
# info = np.random. randint(0,10,size=(2,2,1),dtype=np.uint8)
#
# print("state",state)
#
# test = state[:2,:2,:]
# print("test",test)
#
# print("info",info)
#
# test2 = state[:2,:2,:]+info
# print("test2",test2)

# # 위 코드는 0부터 255 사이의 무작위 정수로 채워진 배열을 생성합니다.
# print(state)
# # 추출하고자 하는 영역의 시작 인덱스를 지정합니다.
# x_start = 1 # x 축 시작 인덱스
# y_start = 2  # y 축 시작 인덱스
# z_start = 0   # z 축 시작 인덱스
#
# # 추출하고자 하는 영역의 크기를 지정합니다.
# x_size = 2  # x 축 크기
# y_size = 2  # y 축 크기
# z_size = 2  # z 축 크기
#
# # 지정한 시작 인덱스와 크기를 사용하여 영역을 추출합니다.
# extracted_area = state[x_start:x_start+x_size, y_start:y_start+y_size, z_start:z_start+z_size]
# print(extracted_area.shape)


# import numpy as np
#
# # 8x8x7 크기의 두 개의 랜덤한 3D 배열 생성
# array1 = np.random.rand(8, 8, 7)
# array2 = np.random.rand(8, 8, 7)
#
# # 두 개의 배열을 수직 방향 (axis=0)으로 이어붙임
# concatenated_array = np.concatenate((array1, array2), axis=0)
#
# # 결과 확인
# print("원본 배열 1:")
# print(array1.shape)  # (8, 8, 7)
# print("원본 배열 2:")
# print(array2.shape)  # (8, 8, 7)
# print("이어붙인 배열:")
# print(concatenated_array.shape)  # (16, 8, 7)
#
#
# def gnn_predict(self, agent, state):  # 에이전트 이름 받아서 해당하는 gnn 네트워크로 값을 뱉어야함
#     state = state.view(-1, dim_feature)  # 예를 들어, (3,3,5)이면 (9,5)꼴로 바꿔주는 작업! graphsage를 구해야 하는 것!
#     idx = get_agent_idx(agent)  # agent의 idx를 가져온다.
#     return self.gsages[idx](state)  # gsages의 forward 함수!이용한다.
#     # 해당 agent에 해당하는 gsage를 가져와서 state 넣어준다. 이때 qsage 는 dqn에 넣을 것과 shared gnn에 넣을 거랑 구분된다.
#     # 이 함수는 2개의 출력값 return dqn, shared 를 받음데
#
#
# ##그런데, 그래프 형태로 나타난 것을 다시 3*3*7의 형태로 바꿀수 있는지 확인해야 할 것 같음. 그래야 45*45*7 에다가 넣지
# def back_to_pixel(self, state, predator_state):
#     return state.view(predator_state)
#
#
# def allocate_to_pixel(self):  # 모서리에 있는 것들은 state의 값들을 어떻게 받아오는지 확인해야 한다.
#     return None
#
#
# def sharing(self, state,
#             pos_predator):  # 결과를 절반은 dqn에 절반은 sigmoid 취해서 shared gnn에 넣는 함수를 만들어야한다. gnn돌린거 넣어주고, dqn에 넣을 shared 부분 가져오고
#     return None
#
#
# def dqn_predict(self, state, shared_state):  # shared gnn 에서 concat하는 부분이 있어야 함
#     return None  # dqn에 넣어서 action을 뽑아내는 코드 여야 함
#
#
# def train(self, states, targets):
#     self.model.fit(states, targets, epochs=1, verbose=0)


import numpy as np


def create_array(shape):
    """
    지정된 형태(shape)의 1부터 1씩 증가하는 숫자로 채워진 NumPy 배열을 생성합니다.

    :param shape: 배열의 형태를 나타내는 튜플 (예: (7, 7, 1))
    :return: 생성된 NumPy 배열
    """
    total_elements = np.prod(shape)  # 배열 요소의 총 개수 계산
    incremental_array = np.arange(1, total_elements + 1).reshape(shape)
    return incremental_array


# 7x7x1 차원의 배열을 생성합니다.
shape = (7, 7, 1)
state = create_array(shape)
#print(state[3:4,3:4,:])


info = create_array((2,2,1))
#print(info)

# 결과 배열 출력





x_start = 1
y_start = 1
z_start = 1

x_size = 2
y_size = 2
z_size = 1

print(state[x_start - x_size-1 :x_start + x_size, y_start - y_size-1 : y_start + y_size,:z_size])


a=[1,2,3,4,5,6,7]
print(a[-1:2])

a=np.random.randint(0,50,size=32)
print(a)
batch_size = 32
indices = range(batch_size)
print(list(indices))

np.random.seed(1)
a=np.random.randint(0,10,(3,4))
print(a)
print(a.max(axis=1))


# def replay(self):  # 업데이트 하는 부분
#     for _ in range(10):
#         states, actions, rewards, next_states, done = self.buffer.sample()  # 위의 생성한 buffer에서 하나의 sample을 뽑음-buffer에는 그냥 model network에서 만든 튜플이 들어가 있음
#         # states 와 next_states 의 32*p 의 꼴이고, action 은 batch size 길이의 일차원 벡터
#         targets = self.target_model.predict(states)  # state를 넣었을 때, 액션에 대한 q value 값이 나올 것이다.
#         # target network으로부터 target을 만들어야 하므로 target에는 32개 각각 10개의 output이 있을 것이므로, 32*10이 되겠지!
#         # 32*10 라는 틀을 만들어 놓는 것!
#         next_q_values = self.target_model.predict(next_states).max(
#             axis=1)  # next state에 대해서 max q value을 예측! batch_size에 들어있는 데이터 next_stae
#         targets[range(args.batch_size), actions] = rewards + (1 - done) * next_q_values * args.gamma
#         self.model.train(states, targets)  # 그렇게 targets 에 대한 y value을 구해서 그에 대해서 train을 진행시킴
#         # state를 넣었을 때의 값과 targets의 값이 같아지도록 업데이트를 해간다.

batch_size = 32

# 가상의 값들
actions = [1, 0, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 3]  # 각 데이터 포인트에서 선택한 액션
rewards = [0.2, -0.1, 0.5, 0.3, -0.2, 0.1, 0.2, 0.4, -0.1, 0.5, 0.3, -0.2, 0.1, 0.2, 0.4, -0.1, 0.5, 0.3, -0.2, 0.1, 0.2, 0.4, -0.1, 0.5, 0.3, -0.2, 0.1, 0.2, 0.4, -0.1, 0.5, 0.7]  # 각 데이터 포인트에서 얻은 보상
done = [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0]  # 각 데이터 포인트에서 게임 종료 여부 (1이면 종료)
next_q_values = np.random.randint(0,10,(32,10))#[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.7]  # 각 데이터 포인트에서의 다음 상태에서의 최대 Q-값
# print("next_q_value",next_q_values.shape)
# print("len",len(actions))
# print(len(next_q_values[0]))
import numpy as np

# 가상의 값들
actions = np.array(actions)
rewards = np.array(rewards)
done = np.array(done)
next_q_values = np.array(next_q_values)

# args 설정
batch_size = 32
gamma = 0.95

# targets 계산
targets = np.zeros((batch_size, len(next_q_values[0])))  # 초기화된 targets 배열 (32x10 크기)
print(targets)
#print("shape",targets.shape)
# targets[range(args.batch_size), actions] 계산

#targets[range(args.batch_size), actions] = rewards + (1 - done) * next_q_values * args.gamma

def plus(self,a):
    return
action_buffer = {i: 0 for i in range(30)}

action_buffer[3]=19
print(action_buffer)


a="predator"
print(a[:8])



import torch

# 두 개의 텐서 생성
tensor1 = torch.randn(3, 4)
tensor2 = torch.randn(2, 4)

# 두 텐서를 연결 (기본적으로 dim=0으로 연결됨)
result = torch.cat((tensor1, tensor2), dim=0)
print(result.shape)


info = torch.zeros((10, 10, 7))  # 원하는 크기로 초기화
shared = torch.zeros((10, 10, 7))  # 원하는 크기로 초기화
input = torch.cat((shared, info), dim=0)
print(input.shape)
