"""
지금 env함수를 깊게 파는 이뉴는 observation을 만드는 last()는 함수의 동작을 알기 위해서이다. 일단 readme를 참고해보면

#### Observation space

The observation space is a 10x10 map for pursuers and a 9x9 map for the pursued. They contain the following channels, which are (in order):

feature | number of channels
--- | ---
obstacle/off the map| 1
my_team_presence| 1
my_team_hp| 1
other_team_presence| 1
other_team_hp| 1
binary_agent_id(extra_features=True)| 10
one_hot_action(extra_features=True)| 9/Prey,13/Predator
last_reward(extra_features=True)| 1

### State space

The observation space is a 45x45 map. It contains the following channels, which are (in order):

feature | number of channels
--- | ---
obstacle map| 1
prey_presence| 1
prey_hp| 1
predator_presence| 1
predator_hp| 1
binary_agent_id(extra_features=True)| 10
one_hot_action(extra_features=True)|  13 (max action space)
last_reward(extra_features=True)| 1


env = make_env(raw_env)
결국 이것도 함수를 만드는 거였음 1. raw_env 라는 함수가 make_env라는 함수의 인자로 들어감 2. make_evn함수에서는 raw_env함수를 어떻게 사용할지 알아야함
def make_env(raw_env):
    def env_fn(**kwargs):
        env = raw_env(**kwargs) #결과값 클래스 class parallel_to_aec_wrapper(AECEnv)의 인스턴스임
        env = wrappers.AssertOutOfBoundsWrapper(env) #그 인스턴스로 뭔가 형 변환을 해서 호환을 맞춰 주는 것인데 이것까지 자세히 알 필요는 없다.
        env = wrappers.OrderEnforcingWrapper(env)
        return env

    return env_fn

return env_fn 인것으로 보아서 make_env함수는 다시 raw_env함수 객체를 return 한다.
make_env함수는 인자로 받은 raw_env함수는 무엇을 할까? **kwargs를 이용한다.
결국 env = make_env(raw_env) 여기로 들어가서 만들어지는 함수에 어떤 인자 K를 넣으면 그것은 env_fn함수에 k 즉, **kwargs으로 들어가 실행된다.
결국...
return parallel_to_aec_wrapper(
        parallel_env(map_size, max_cycles, minimap_mode, extra_features, **reward_args))
이것을 실행하는 것이다. 그리고 env_fn함수안에 wrappers.AssertOutOfBoundsWrappe,wrappers.OrderEnforcingWrapper 이 함수는 그냥 뭔가 호한을 위한 함수인것 같다..알 필요는 굳이...
다만 raw_env(**kwargs)함수는 중요한 것 같다. 그래서  결국 parallel_to_aec_wrapper 이 함수가 중요하다는 것이고, 이 함수를 확인해야 한다.
그런데 마침 test파일에 observation을 만드는 함수가 env.last()인데, 이 last함수가 class parallel_to_aec_wrapper(AECEnv) 이 클래스 안에 있다.
env.py 파일안에 AECEnv 클래스가 있다. 그리고 그 클래스 안에 last함수가 있다.
지금 목적은
env.state().shape
(45, 45, 9)
observation.shape
(10, 10, 9)
왜 이렇게 되는지 아는 것이다. 각 channel이 의미하는 바가 무엇인지...그런데...굳이 또 알필요는 없나? 아니 알아야 한다. 각 픽셀의 절대적인 위치를 알아야 gnn에 넣을 수 있다.

또한 채널 아는 것보다, 각 에이전트의 절대 좌표의 값을 알아야 그래프를 구성해서 넣어서 업데이트를 진행할 수 있는데, 그건 minimap_mode에서 확인할 수 있을 것 같다. 문제는 minimap_mode의
agent_position이 0과 1사이의 값으로 표준화 되어서 나오는게 문제인데... 이것만 수정하면 절댓값을 얻을 수 있을 것
minimap_mode 사용되는 쪽으로 가보니까 magent_env 파일에 86라인 보면 인덱스 가져와서 normalize하는거 볼 수 있음


walls = self.env._get_walls_info() 이 부분에서 wall들의 위치를 알 수 있는 것 같은데..(근데 이놈들의 좌표는 필요가 없을 것 같긴 하다 아마도 feature들에 다 있지 않을까 싶다) magent_env파일에 있\

 if len(self.handles) == 3:
            vs = " vs ", (0, 0, 0)
            result = [(form_txt(0), vs, form_txt(1))]
render에 get banner함수에서 집단이 3개일때는 정의가 안되어 잇는데...이거 고치면...되는 건가

def _compute_terminates(self, step_done): 이건?
"""
#
#
# import torch
#
# # 기존 텐서 생성
# x = torch.tensor([[1, 2, 3],
#                   [4, 5, 6]])
#
# # view() 메서드를 사용하여 모양 변경
# x = x.view(3, 2)  # x의 모양을 (3, 2)로 변경
#
# # 또는 reshape() 메서드를 사용하여 모양 변경
# #x = x.reshape(2, 3)  # x의 모양을 (2, 3)으로 변경
#
# print(x)
#

import torch as th
# num_agents= 20
# observations = [None] * num_agents  # 에이전트 수만큼 None으로 초기화된 리스트 생성
#
# print(observations)
#
#
# entire_state = (10,3)
# #에이전트 0의 observation을 저장
# observations[0] = th.zeros(entire_state)
# observations[0].append(th.zeros(entire_state))
#
# # 에이전트 1의 observation을 저장
# observations[1] = th.zeros(entire_state)
#
# print(observations)
#
# # 에이전트 0의 observation을 얻기
# #agent0_observation = observations[0]


# import torch as th
#
# num_agents = 20
# num_steps = 100  # 예를 들어, 100개의 step에 대한 에이전트 observation 저장
#
# # 에이전트 수와 스텝 수를 고려한 3D Tensor를 초기화합니다.
# observations = th.zeros((num_agents, num_steps, 10, 10, 3))
# print(observations.shape)
#
# # 각 스텝에서 에이전트 observation을 업데이트합니다.
# for step in range(num_steps):
#     # 각 에이전트로부터 observation을 받아서 업데이트합니다.
#     for agent_idx in range(num_agents):
#         # 에이전트로부터 observation을 받는 로직 (예: 환경에서 observation을 받아오는 부분)
#         agent_observation = th.randn(10, 10, 3)  # 임의의 예시 데이터
#
#         # 에이전트의 observation을 저장합니다.
#         observations[agent_idx, step] = agent_observation
#
# # 예시로 첫 번째 에이전트의 첫 번째 스텝의 observation을 출력합니다.
#
#
# print(observations[0, 0])


# import torch as th
#
# num_agents = 20
# num_steps = 100  # 예를 들어, 100개의 step에 대한 에이전트 observation 저장
#
# # 딕셔너리 초기화
# observations_dict = {}
#
# # 각 에이전트에 대한 딕셔너리 초기화
# for agent_idx in range(num_agents):
#     agent_key = f"agent_{agent_idx}"
#     observations_dict[agent_key] = []
#
# # 각 스텝에서 에이전트 observation을 업데이트합니다.
# for step in range(num_steps):
#     # 각 에이전트로부터 observation을 받아서 업데이트합니다.
#     for agent_idx in range(num_agents):
#         agent_key = f"agent_{agent_idx}"
#
#         # 에이전트로부터 observation을 받는 로직 (예: 환경에서 observation을 받아오는 부분)
#         agent_observation = th.randn(3*4)  # 임의의 예시 데이터
#
#         # 에이전트의 observation을 해당 키에 저장합니다.
#         observations_dict[agent_key].append(agent_observation)
#
# # 예시로 첫 번째 에이전트의 첫 번째 스텝의 observation을 출력합니다.
# first_agent_key = "agent_0"
# print(observations_dict[first_agent_key][0])

#
# import torch as th
#
# num_agents = 20
# num_steps = 100  # 예를 들어, 100개의 step에 대한 에이전트 observation 저장
#
# # 딕셔너리 초기화
# observations_dict = {}
#
# # 각 에이전트에 대한 딕셔너리 초기화
# for agent_idx in range(num_agents):
#     agent_key = f"agent_{agent_idx}"
#     observations_dict[agent_key] = []
#
# # 각 스텝에서 에이전트 observation을 업데이트합니다.
# for step in range(num_steps):
#     # 각 에이전트로부터 observation을 받아서 업데이트합니다.
#     for agent_idx in range(num_agents):
#         agent_key = f"agent_{agent_idx}"
#
#         # 에이전트로부터 observation을 받는 로직 (예: 환경에서 observation을 받아오는 부분)
#         agent_observation = th.randn((3,4))  # 임의의 예시 데이터
#
#         # 에이전트의 observation을 해당 키에 저장합니다.
#         observations_dict[agent_key].append(agent_observation)
#
# # 예시로 첫 번째 에이전트의 첫 번째 스텝의 observation을 출력합니다.
#
# first_agent_key = "agent_0"
# print(observations_dict[first_agent_key][3].shape)

# reward_dict 예시 (20개의 키-값 쌍)
# reward_dict = {
#     'agent_0': [1, 2, 3, 4, 5],
#     'agent_1': [2, 3, 4, 5, 6],
#     # 나머지 18개 에이전트의 데이터도 포함
# }
#
# # 각 리스트의 마지막 값을 더할 변수 초기화
# total_last_rewards = 0
#
# # 각 리스트의 마지막 값을 더하기
# for agent_rewards in reward_dict.values():
#     print(agent_rewards)
#     if len(agent_rewards) > 0:
#         last_reward = agent_rewards[-1]
#         total_last_rewards += last_reward
#
# # 각 리스트의 마지막 값들을 더한 결과 출력
# print("각 리스트의 마지막 값들을 더한 결과:", total_last_rewards)
#


dividend = 10
divisor = 3
quotient =(29// 30)
print(quotient)  # 출력: 3
