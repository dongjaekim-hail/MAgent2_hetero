from magent2.environments import hetero_adversarial_v1
from MADQN2 import MADQN
from arguments import args
import argparse
import numpy as np
import torch as th
import wandb


# wandb.init(project="MADQN", entity='junhyeon')
# wandb.run.name = 'test1'




parser = argparse.ArgumentParser()
parser.add_argument('--g'
					'amma', type=float, default=0.95)
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
n_prey = 10
predator1_adj = (100,100)
predator2_adj = (36,36)


batch_size = 1
predator1_view_range = 5
predator2_view_range = 3


shared = th.zeros(entire_state)
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

	# observation 딕셔너리 초기화
	observations_dict = {}
	# 각 에이전트에 대한 딕셔너리 초기화
	for agent_idx in range(n_predator1 + n_predator2):
		observations_dict[agent_idx] = []

	# # next_observation 딕셔너리 초기화
	# next_observations_dict = {}
	# # 각 에이전트에 대한 딕셔너리 초기화
	# for agent_idx in range(n_predator1 + n_predator2):
	# 	next_observations_dict[agent_idx] = []

	# reward 딕셔너리 초기화
	reward_dict = {}
	# 각 에이전트에 대한 딕셔너리 초기화
	for agent_idx in range(n_predator1 + n_predator2):
		reward_dict[agent_idx] = []

	# action 딕셔너리 초기화
	action_dict = {}
	# 각 에이전트에 대한 딕셔너리 초기화
	for agent_idx in range(n_predator1 + n_predator2):
		action_dict[agent_idx] = []

	# termination 딕셔너리 초기화
	termination_dict = {}
	# 각 에이전트에 대한 딕셔너리 초기화
	for agent_idx in range(n_predator1 + n_predator2):
		termination_dict[agent_idx] = []

	#truncation 초기화
	truncation_dict = {}
	# 각 에이전트에 대한 딕셔너리 초기화
	for agent_idx in range(n_predator1 + n_predator2):
		truncation_dict[agent_idx] = []


	iteration_number = 0


	##########################################################################
	##env.last() 할때마다 다음 agent의 상태정보를 가져오기에 for문을 두번 연달아서 해야한다.##
	##########################################################################


	for agent in env.agent_iter():
		if iteration_number // (n_predator1 + n_predator2 + n_prey) == 0:   #첫번째 step

			print("iteration",(iteration_number // (n_predator1 + n_predator2 + n_prey)) % 2)
			print("iteration_numbber",iteration_number)

			if agent[:8] == "predator": #predator 일때만 밑의 식이 실행되어야 함

				#잡단에 있는 predator들의 절대 좌표
				handles = env.env.env.env.env.get_handles()
				pos_predator1 = env.env.env.env.env.get_pos(handles[0])
				pos_predator2 = env.env.env.env.env.get_pos(handles[1])

				#에이전트 자신에게 맞는 pos 가져오는 부분
				if agent[9] == "1":  # 1번째 predator 집단
					idx = int(agent[11:])
					#print("에이전트idx#################################################################",idx)
					pos = pos_predator1[idx]
					view_range = predator1_view_range
				else:				# 2번째 predator 집단
					idx = int(agent[11:]) + n_predator1
					#print("에이전트idx#################################################################",idx)
					pos = pos_predator2[idx - n_predator1]
					view_range = predator2_view_range

				print(idx,":idx")

				# 이 함수를 통해 현재 돌고 있는 agent에 맞는 idx, adj, pos, view_range, gdqn, target_gdqn, buffer등을 설정해준다.
				madqn.set_agent_info(agent, pos, view_range)



				#  현재 observation를 받아오기 위한 것
				observation, reward, termination, truncation, info, agent = env.last() #애초에 reward가 누적보상값이네 이거 수정이 필요하다.
				#print(" 첫번째 last 가 반환하는 agent 의 정보",agent)
				observation_temp = process_array(observation)
				action = madqn.get_action(state=observation_temp, mask=None)
				env.step(action)

				observations_dict[idx].append(observation_temp)
				action_dict[idx].append(action) #굳이 action 은 저장하지 않아도 됨!
				reward_dict[idx].append(reward)
				termination_dict[idx].append(termination)
				truncation_dict[idx].append(truncation)



			else: #prey들은 별도의 절차없이 action 을 선택하고 step을 진행해 나간다.

				action = env.action_space(agent).sample()
				env.step(action)

		else: #두번째 step 이후

			step_idx= iteration_number // 30
			print(step_idx,":step_idx")

			if agent[:8] == "predator":  # predator 일때만 밑의 식이 실행되어야 함

				# 잡단에 있는 predator들의 절대 좌표
				handles = env.env.env.env.env.get_handles()
				pos_predator1 = env.env.env.env.env.get_pos(handles[0])
				pos_predator2 = env.env.env.env.env.get_pos(handles[1])

				# 에이전트 자신에게 맞는 pos 가져오는 부분
				if agent[9] == "1":  # 1번째 predator 집단
					idx = int(agent[11:])
					#print("에이전트idx#################################################################", idx)
					pos = pos_predator1[idx]
					view_range = predator1_view_range
				else:  # 2번째 predator 집단
					idx = int(agent[11:]) + n_predator1
					#print("에이전트idx#################################################################", idx)
					pos = pos_predator2[idx - n_predator1]
					view_range = predator2_view_range

				print(idx, ":idx")

				# 이 함수를 통해 현재 돌고 있는 agent에 맞는 idx, adj, pos, view_range, gdqn, target_gdqn, buffer등을 설정해준다.
				madqn.set_agent_info(agent, pos, view_range)

				#  현재 observation를 받아오기 위한 것
				observation, reward, termination, truncation, info, agent = env.last() #애초에 reward가 누적보상값이네 이거 수정이 필요하다.
				#print(" 첫번째 last 가 반환하는 agent 의 정보", agent)
				observation_temp = process_array(observation)

				if termination or truncation:
					print(agent , 'is terminated')
					env.step(None)
					continue

				else:
					action = madqn.get_action(state=observation_temp, mask=None)
					env.step(action)

					observations_dict[idx].append(observation_temp)
					action_dict[idx].append(action)  # 굳이 action 은 저장하지 않아도 됨!
					reward_dict[idx].append(reward)
					termination_dict[idx].append(termination)
					truncation_dict[idx].append(truncation)

					print(len(observations_dict[idx]))
					print(len(action_dict[idx]))
					print(len(reward_dict[idx]))
					print(len(termination_dict[idx]))
					print(len(truncation_dict[idx]))
					madqn.buffer.put(observations_dict[idx][step_idx-1], action_dict[idx][step_idx], reward_dict[idx][step_idx]-reward_dict[idx][step_idx-1],
									 observations_dict[idx][step_idx], termination_dict[idx][step_idx], truncation_dict[idx][step_idx])



				# 히스토리가 batchsize 보다 넘게 쌓였으면 업데이트를 진행한다.
				if madqn.buffer.size() >= args.batch_size:
				#if madqn.buffer.size() >= 10:
					print("replay")
					madqn.replay()
					madqn.target_update()
					#print('EP{} EpisodeReward={}'.format(ep, [idx]))


			else:  # prey들은 별도의 절차없이 action 을 선택하고 step을 진행해 나간다.

				action = env.action_space(agent).sample()
				env.step(action)

		iteration_number += 1

		# 어차피 reward_dict의 각 에이전트의 리스트에 담겨있는 reward 값을 누적보상값이므로, 각각의 에이전트의 reward 리스트 마지막것들만 더하면 predator의 총 reward 값이다.
		total_last_rewards = 0

		# 각 리스트의 마지막 값을 더하기
		for agent_rewards in reward_dict.values():

			if len(agent_rewards) == 0:
				print("first step")

			elif len(agent_rewards) > 0 or len(agent_rewards) == 1:
				last_reward = agent_rewards[-1]
				total_last_rewards += last_reward

			else:
				last_reward = agent_rewards[-1] - agent_rewards[-2]
				total_last_rewards += last_reward

		# 각 리스트의 마지막 값들을 더한 결과 출력
		print("predator팀의 전체 reward", total_last_rewards)
		#wandb.log({"total_last_rewards": total_last_rewards})

# env.state() # receives the entire state



