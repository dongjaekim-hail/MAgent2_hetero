from magent2.environments import hetero_adversarial_v1
from MADQN_cpu_cen import MADQN
from arguments import args
import argparse
import numpy as np
import torch as th
import wandb

parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.95)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--eps', type=float, default=1.0)
parser.add_argument('--eps_decay', type=float, default=0.995)
parser.add_argument('--eps_min', type=float, default=0.01)
parser.add_argument('--max_update_steps', type=int, default=100)
parser.add_argument('--total_ep', type=int, default=10000)
parser.add_argument('--info_decay',type=int, default=0.5)
parser.add_argument('--buffer_size', type=int, default=50000)
parser.add_argument('--trainstart_buffersize', type=int, default=10000)
parser.add_argument('--book_decay',type=int, default=0.8)

device = 'cpu'

args = parser.parse_args()  #Namespace(gamma=0.95, lr=0.005, batch_size=32, eps=1.0, eps_decay=0.995, eps_min=0.01)

# wandb.init(project="MADQN", entity='hails',config=args.__dict__)
# wandb.run.name = 'semi9_cpu_mapsize=25'


render_mode = 'rgb_array'
env = hetero_adversarial_v1.env(map_size=25, minimap_mode=False, tag_penalty=-0.2,
								max_cycles=args.max_update_steps, extra_features=False,render_mode=render_mode)

entire_state = (25,25,3)
dim_act = 13
n_predator1 = 10
n_predator2 = 10
n_prey = 10
predator1_adj = (625,625)
predator2_adj = (625,625)


batch_size = 1
predator1_view_range = 5
predator2_view_range = 3


#shared = th.zeros(entire_state)
madqn = MADQN(n_predator1, n_predator2, dim_act ,entire_state, device, buffer_size=args.buffer_size)

def process_array(arr):

    arr = np.delete(arr, [2, 4, 6], axis=2)
    combined_dim = np.logical_or(arr[:, :, 2], arr[:, :, 3])
    result = np.dstack((arr[:, :, :2], combined_dim))

    return result


def main():
	for ep in range(1000000):
		ep_reward = 0
		env = hetero_adversarial_v1.env(map_size=25, minimap_mode=False, tag_penalty=-0.2,
										max_cycles=args.max_update_steps, extra_features=False, render_mode=render_mode)

		env.reset()
		print("ep:",ep,'*' * 80)
		iteration_number = 0

		observations_dict = {}
		for agent_idx in range(n_predator1 + n_predator2):
			observations_dict[agent_idx] = []

		reward_dict = {}
		for agent_idx in range(n_predator1 + n_predator2):
			reward_dict[agent_idx] = []

		action_dict = {}
		for agent_idx in range(n_predator1 + n_predator2):
			action_dict[agent_idx] = []

		termination_dict = {}
		for agent_idx in range(n_predator1 + n_predator2):
			termination_dict[agent_idx] = []

		truncation_dict = {}
		for agent_idx in range(n_predator1 + n_predator2):
			truncation_dict[agent_idx] = []



		for agent in env.agent_iter():

			step_idx = iteration_number // 30

			if step_idx == 0:

				if agent[:8] == "predator":

					#idx = int(agent[11:])

					if agent[9] == "1":
						idx = int(agent[11:])

					else:
						idx = int(agent[11:]) + n_predator1


					madqn.set_agent_info(agent)


					_ , reward, termination, truncation, info  = env.last()
					observation = env.state()
					observation_temp = process_array(observation)
					action = madqn.get_action(state=observation_temp, mask=None )
					env.step(action)

					observations_dict[idx].append(observation_temp)
					action_dict[idx].append(action)
					reward_dict[idx].append(reward)



				else: #prey 일때

					action = env.action_space(agent).sample()
					env.step(action)

			else: #두번째 step 이후


				if agent[:8] == "predator":

					#idx = int(agent[11:])
					if agent[9] == "1":
						idx = int(agent[11:])

					else:
						idx = int(agent[11:]) + n_predator1


					madqn.set_agent_info(agent)


					_ , reward, termination, truncation, info = env.last()
					observation = env.state()
					observation_temp = process_array(observation)

					if termination or truncation:
						print(agent , 'is terminated')
						env.step(None)
						continue

					else:
						action = madqn.get_action(state=observation_temp, mask=None)
						env.step(action)

						observations_dict[idx].append(observation_temp)
						action_dict[idx].append(action)
						reward_dict[idx].append(reward)
						termination_dict[idx].append(termination)
						truncation_dict[idx].append(truncation)


						if madqn.buffer.size() >= args.trainstart_buffersize:

							madqn.replay()
							madqn.target_update()

						# 총 1000000번의 iteration 중에 10000번 돌고서 target_update를 진행한다.
						if iteration_number % 10000 == 0:
							madqn.target_update()



				else:  #두번째 step 이후 prey
					observation, reward, termination, truncation, info = env.last()

					if termination or truncation:
						print(agent, 'is terminated')
						env.step(None)
						continue

					else:
						action = env.action_space(agent).sample()
						env.step(0)




			if (iteration_number + 1) % 30 == 0 and iteration_number > 30:

				total_last_rewards = 0
				for agent_rewards in reward_dict.values():


					last_reward = agent_rewards[-1] - agent_rewards[-2]
					total_last_rewards += last_reward

				ep_reward += total_last_rewards

				for idx in range(n_predator1 + n_predator2):

					# madqn.set_agent_info(agent, pos, view_range)

					madqn.set_agent_buffer(idx)

					madqn.buffer.put(observations_dict[idx][step_idx - 1],
									 action_dict[idx][step_idx - 1],
									 total_last_rewards,
									 observations_dict[idx][step_idx],
									 termination_dict[idx][step_idx - 1],
									 truncation_dict[idx][step_idx - 1])


				ep_reward += total_last_rewards
				print("predator_total_reward", total_last_rewards)
				#wandb.log({"total_last_rewards": total_last_rewards })

			iteration_number += 1
		#wandb.log({"ep_reward": ep_reward})


		#ep_reward += total_last_rewards
		#print("ep_reward:", ep_reward)

		# if iteration_number > args.max_update_steps:
		# 	print('*' * 10, 'train over', '*' * 10)
		# 	print(iteration_number)
		# 	break


		if ep > args.total_ep: #100
			print('*' * 10, 'train over', '*' * 10)
			print(iteration_number)
			break

		# if (ep % 1000) ==0 :
		# 	for i in range(len(madqn.gdqns)) :
		# 		th.save(madqn.gdqns[i].state_dict(), 'model_save/'+'model_'+ str(i) + '_ep' +str(ep) +'.pt')
		# 		th.save(madqn.gdqns[i].state_dict(), 'model_save/' + 'model_target_' + str(i) + '_ep' + str(ep)+ '.pt')

	print('*' * 10, 'train over', '*' * 10)
	#print(iteration_number)
	# env.state() # receives the entire state

if __name__ == '__main__':
	main()
	#print('done')
	#??? ??
	# for i in range(len(madqn.gdqns)) :
	# 	th.save(madqn.gdqns[i].state_dict(), 'model_save/'+'model_'+ str(i) +'.pt')
	# 	th.save(madqn.gdqns[i].state_dict(), 'model_save/' + 'model_target_' + str(i) + '.pt')


	print('done')
