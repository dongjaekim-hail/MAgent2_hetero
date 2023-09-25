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
state = np.random.randint(0, 50, size=(5, 5, 1), dtype=np.uint8)
# 위 코드는 0부터 255 사이의 무작위 정수로 채워진 배열을 생성합니다.
print(state)
# 추출하고자 하는 영역의 시작 인덱스를 지정합니다.
x_start = 1 # x 축 시작 인덱스
y_start = 2  # y 축 시작 인덱스
z_start = 0   # z 축 시작 인덱스

# 추출하고자 하는 영역의 크기를 지정합니다.
x_size = 2  # x 축 크기
y_size = 2  # y 축 크기
z_size = 2  # z 축 크기

# 지정한 시작 인덱스와 크기를 사용하여 영역을 추출합니다.
extracted_area = state[x_start:x_start+x_size, y_start:y_start+y_size, z_start:z_start+z_size]
print(extracted_area.shape)


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
