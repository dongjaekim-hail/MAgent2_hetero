from magent2.environments import hetero_adversarial_v1
from Model.MADQN import MADQN, args
import argparse
import numpy as np
import torch as th
import wandb


device = 'cpu'



# wandb.init(project="MADQN", entity='hails',config=args.__dict__)
# wandb.run.name = 'semi10_cpu_mapsize=25'

render_mode = 'rgb_array'
env = hetero_adversarial_v1.env(map_size=args.map_size, minimap_mode=False, tag_penalty=-0.2,
								max_cycles=args.max_update_steps, extra_features=False,render_mode=render_mode)

predator1_view_range = args.predator1_view_range
predator2_view_range = args.predator2_view_range
n_predator1 = args.n_predator1
n_predator2 = args.n_predator2
n_prey = args.n_prey

shared_shape = (args.map_size + predator1_view_range*2 ,args.map_size + predator1_view_range*2,3)
predator1_obs = (predator1_view_range*2,predator1_view_range*2,3)
predator2_obs = (predator2_view_range*2,predator2_view_range*2,3)
dim_act = 13

predator1_adj = ((predator1_view_range*2)^2,(predator1_view_range*2)^2)
predator2_adj = ((predator2_view_range*2)^2,(predator2_view_range*2)^2)

batch_size = 1


shared = th.zeros(shared_shape)
madqn = MADQN(n_predator1, n_predator2, predator1_obs, predator2_obs, dim_act ,shared_shape, shared, device, buffer_size=args.buffer_size)

def process_array(arr):

    arr = np.delete(arr, [2, 4, 6], axis=2)

    combined_dim = np.logical_or(arr[:, :, 2], arr[:, :, 3])

    result = np.dstack((arr[:, :, :2], combined_dim))

    return result


def main():
	for ep in range(1000000):
		ep_reward = 0

		env = hetero_adversarial_v1.env(map_size=args.map_size, minimap_mode=False, tag_penalty=-0.2,
										max_cycles=args.max_update_steps, extra_features=False, render_mode=render_mode)

		madqn.reset_shred(shared) #? ?????? shared reset
		env.reset()
		print("ep:",ep,'*' * 80)

		# observation ???? ???
		observations_dict = {}
		# ? ????? ?? ???? ???
		for agent_idx in range(n_predator1 + n_predator2):
			observations_dict[agent_idx] = []

		# # next_observation ???? ???
		# next_observations_dict = {}
		# # ? ????? ?? ???? ???
		# for agent_idx in range(n_predator1 + n_predator2):
		# 	next_observations_dict[agent_idx] = []

		# reward ???? ???
		reward_dict = {}
		# ? ????? ?? ???? ???
		for agent_idx in range(n_predator1 + n_predator2):
			reward_dict[agent_idx] = []

		# action ???? ???
		action_dict = {}
		# ? ????? ?? ???? ???
		for agent_idx in range(n_predator1 + n_predator2):
			action_dict[agent_idx] = []

		# termination ???? ???
		termination_dict = {}
		# ? ????? ?? ???? ???
		for agent_idx in range(n_predator1 + n_predator2):
			termination_dict[agent_idx] = []

		#truncation ???
		truncation_dict = {}
		# ? ????? ?? ???? ???
		for agent_idx in range(n_predator1 + n_predator2):
			truncation_dict[agent_idx] = []

		# book  ???
		book_dict = {}
		# ? ????? ?? ???? ???
		for agent_idx in range(n_predator1 + n_predator2):
			book_dict[agent_idx] = []

		iteration_number = 0


		##########################################################################
		##env.last() ???? ?? agent? ????? ????? for?? ?? ???? ????.##
		##########################################################################


		for agent in env.agent_iter():

			step_idx = iteration_number // 30
			# if step_idx == 100: #step_idx=100 ?? ?? ep?
			# 	print('ah')
			# 	break

			if step_idx == 0:   #첫번째 step

				# print(iteration_number,"step_idx ???")
				# print(step_idx,"step_idx")

				if agent[:8] == "predator":

					#??? ?? predator?? ?? ??
					handles = env.env.env.env.env.get_handles()
					pos_predator1 = env.env.env.env.env.get_pos(handles[0])
					pos_predator2 = env.env.env.env.env.get_pos(handles[1])

					#???? ???? ?? pos ???? ??
					if agent[9] == "1":
						idx = int(agent[11:])

						pos = pos_predator1[idx]
						view_range = predator1_view_range
					else:
						idx = int(agent[11:]) + n_predator1

						pos = pos_predator2[idx - n_predator1]
						view_range = predator2_view_range




					madqn.set_agent_info(agent, pos, view_range)




					observation, reward, termination, truncation, info  = env.last()

					observation_temp = process_array(observation)
					action, book = madqn.get_action(state=observation_temp, mask=None)
					env.step(action)

					observations_dict[idx].append(observation_temp)
					action_dict[idx].append(action)
					reward_dict[idx].append(reward)
					book_dict[idx].append(book)



				else:

					action = env.action_space(agent).sample()
					env.step(0)

			else: #두번째 step 이후


				if agent[:8] == "predator":


						handles = env.env.env.env.env.get_handles()
						pos_predator1 = env.env.env.env.env.get_pos(handles[0])
						pos_predator2 = env.env.env.env.env.get_pos(handles[1])


						if agent[9] == "1":
							idx = int(agent[11:])
							pos = pos_predator1[idx]
							view_range = predator1_view_range
						else:  # 2?? predator ??
							idx = int(agent[11:]) + n_predator1
							pos = pos_predator2[idx - n_predator1]
							view_range = predator2_view_range

						madqn.set_agent_info(agent, pos, view_range)


						observation, reward, termination, truncation, info = env.last()

						observation_temp = process_array(observation)



						if termination or truncation:
							print(agent , 'is terminated')
							env.step(None)
							continue

						else:
							action, book  = madqn.get_action(state=observation_temp, mask=None)
							env.step(action)

							observations_dict[idx].append(observation_temp)
							action_dict[idx].append(action)
							reward_dict[idx].append(reward)
							book_dict[idx].append(book)
							termination_dict[idx].append(termination)
							truncation_dict[idx].append(truncation)



						if madqn.buffer.size() >= args.trainstart_buffersize:
							madqn.replay()

						#총 1000000번의 iteration 중에 10000번 돌고서 target_update를 진행한다.
						if iteration_number % 10000 == 0 :
							madqn.target_update()



				else:  # prey?? ??? ???? action ? ???? step? ??? ???.
					observation, reward, termination, truncation, info = env.last()

					if termination or truncation:
						print(agent, 'is terminated')
						env.step(None)
						continue

					else:

						action = env.action_space(agent).sample()
						env.step(0)
						#print(agent,"prey")

			if (iteration_number + 1) % 30 == 0 and iteration_number > 30: #두번째 step 이후, 각 에이전트의 iteraiton이 끝난 직후

				madqn.shared_decay() #shared 에 있는 정보를 decaying 해준다.

				total_last_rewards = 0
				for agent_rewards in reward_dict.values(): #20개의 각 에이전트의 reward 리스트에서 가장 최근의 reward를 빼온다.

					last_reward = agent_rewards[-1] - agent_rewards[-2]
					total_last_rewards += last_reward

				ep_reward += total_last_rewards



				#wandb.log({"total_last_rewards": total_last_rewards })
				print("predator_total_reward", total_last_rewards)

				for idx in range(n_predator1 + n_predator2):
					#특정 한 step에서의 마지막에 buffer를 채우는 작업을 하는 이유 : 특정 action 에 대해 받는 reward가 전체 에이전트 reward의 합이기 때문

					# madqn.set_agent_info(agent, pos, view_range)

					madqn.set_agent_buffer(idx)

					madqn.buffer.put(observations_dict[idx][step_idx - 1], book_dict[idx][step_idx - 1],
									 action_dict[idx][step_idx - 1],
									 total_last_rewards,
									 observations_dict[idx][step_idx], book_dict[idx][step_idx],
									 termination_dict[idx][step_idx-1 ],
									 truncation_dict[idx][step_idx-1 ])


			iteration_number += 1
		#wandb.log({"ep_reward": ep_reward})



		#print("ep_reward:", ep_reward)

		# if iteration_number > args.max_update_steps:
		# 	print('*' * 10, 'train over', '*' * 10)
		# 	print(iteration_number)
		# 	break


		if ep > args.total_ep: #100
			print('*' * 10, 'train over', '*' * 10)
			print(iteration_number)
			break

		if (ep % 1000) ==0 :
			for i in range(len(madqn.gdqns)) :
				th.save(madqn.gdqns[i].state_dict(), 'model_save/'+'model_'+ str(i) + '_ep' +str(ep) +'.pt')
				th.save(madqn.gdqns[i].state_dict(), 'model_save/' + 'model_target_' + str(i) + '_ep' + str(ep)+ '.pt')

	print('*' * 10, 'train over', '*' * 10)
	print(iteration_number)
	# env.state() # receives the entire state

if __name__ == '__main__':
	main()
	#print('done')
	#??? ??
	for i in range(len(madqn.gdqns)) :
		th.save(madqn.gdqns[i].state_dict(), 'model_save/'+'model_'+ str(i) +'.pt')
		th.save(madqn.gdqns[i].state_dict(), 'model_save/' + 'model_target_' + str(i) + '.pt')


	print('done')


