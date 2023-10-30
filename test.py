from magent2.environments import hetero_adversarial_v1
from MADQN_junhyeon import MADQN
from arguments import args
import argparse
import numpy as np
import torch as th

th.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.95)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--eps', type=float, default=1.0)
parser.add_argument('--eps_decay', type=float, default=0.995)
parser.add_argument('--eps_min', type=float, default=0.01)

args = parser.parse_args()  #Namespace(gamma=0.95, lr=0.005, batch_size=32, eps=1.0, eps_decay=0.995, eps_min=0.01)

render_mode = 'rgb_array'
env = hetero_adversarial_v1.env(map_size=45, minimap_mode=False, tag_penalty=-0.2,
								max_cycles=50000, extra_features=False,render_mode=render_mode)

entire_state = (65,65,3)
predator1_obs = (10,10,3)
predator2_obs = (6,6,3)
dim_act = 13
n_predator1 = 10
n_predator2 = 10

batch_size = 1
predator1_view_range = 5
predator2_view_range = 3


shared = th.zeros(entire_state)
shared.detach()
madqn = MADQN(n_predator1, n_predator2, predator1_obs, predator2_obs, dim_act ,entire_state,shared)

def process_array(arr):
    # 3번째, 5번째, 7번째 차원 삭제
    arr = np.delete(arr, [2, 4, 6], axis=2)

    # 4번째와 6번째 차원을 OR 연산하여 하나로 묶기
    combined_dim = np.logical_or(arr[:, :, 2], arr[:, :, 3])

    # 결과 배열 생성 (10, 10, 3)
    result = np.dstack((arr[:, :, :2], combined_dim))

    return result



for ep in range(1000):
	env.reset()
	print("ep:",ep)
	#total_reward 계산하기 위한 predator 길이의 딕셔너리 생성
	total_reward = {i:0 for i in range(n_predator1+n_predator2)}




	for agent in env.agent_iter():

		if agent[:8] == "predator": #predator 일때만 밑의 식이 실행되어야 함
			#잡단에 있는 predator들의 절대 좌표
			handles = env.env.env.env.env.get_handles()
			pos_predator1 = env.env.env.env.env.get_pos(handles[0])
			pos_predator2 = env.env.env.env.env.get_pos(handles[1])

			#에이전트 자신에게 맞는 pos 가져오는 부분
			if agent[9] == "1":  # 1번째 predator 집단
				idx = int(agent[11:])
				pos = pos_predator1[idx]
				view_range = predator1_view_range
			else:				# 2번째 predator 집단
				idx = int(agent[11:])
				pos = pos_predator2[idx]
				view_range = predator2_view_range

			# 이 함수를 통해 현재 돌고 있는 agent에 맞는 idx, adj, pos, view_range, gdqn, target_gdqn, buffer등을 설정해준다.
			madqn.set_agent_info(agent, pos, view_range)

		

			#  현재 observation를 받아오기 위한 것
			observation, reward, termination, truncation, info = env.last()
			observation_temp = process_array(observation)


			#action을 뽑고, 그 액션으로 step을 진행한다.

			if termination or truncation:
				print(agent, ' is terminated')
				env.step(None) # need this

				continue
			else:

				action = madqn.get_action(state = observation_temp, mask=None)
				env.step(action)
				next_observation, reward, termination, truncation, info = env.last()
				madqn.buffer.put(observation_temp, action, reward, process_array(next_observation), termination, truncation)
				total_reward[idx] += reward

			# 히스토리가 batchsize 보다 넘게 쌓였으면 업데이트를 진행한다.
			if madqn.buffer.size() >= args.batch_size:
				madqn.replay()
				madqn.target_update()
				print('EP{} EpisodeReward={}'.format(ep, total_reward[idx]))
				#reward의 위치가 여기가 많나...맞는 듯!


		else: #prey들은 별도의 절차없이 action 을 선택하고 step을 진행해 나간다.
			action = env.action_space(agent).sample()
			env.step(action)



	#env.state() # receives the entire state