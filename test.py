from magent2.environments import hetero_adversarial_v1
#import MADQN
import arguments
import argparse
import numpy as np
#from magent2.environments import adversarial_pursuit_v4


parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.95)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--eps', type=float, default=1.0)
parser.add_argument('--eps_decay', type=float, default=0.995)
parser.add_argument('--eps_min', type=float, default=0.01)

args = parser.parse_args()  #Namespace(gamma=0.95, lr=0.005, batch_size=32, eps=1.0, eps_decay=0.995, eps_min=0.01)

entire_state = (45,45,7)
predator1_obs = (8,8,7)
predator2_obs = (10,10,7)
dim_act = 13
n_predator1 = 10
n_predator2 = 10
batch_size = 10
shared = np.zeros(entire_state)

render_mode = 'human'
env = hetero_adversarial_v1.env(map_size=45, minimap_mode=False, tag_penalty=-0.2,
								max_cycles=50000, extra_features=False,render_mode=render_mode)

#MADQN = MADQN(n_predator1, n_predator2, predator1_obs, predator2_obs, dim_act ,shared)


for ep in range(1000):
	env.reset()
	print("ep:",ep)

	for agent in env.agent_iter():
		handles = env.env.env.env.env.get_handles()
		pos_predator1 = env.env.env.env.env.get_pos(handles[0])
		pos_predator2 = env.env.env.env.env.get_pos(handles[1])
		#prey1 = env.env.env.env.env.get_pos(handles[2])

		print(pos_predator1)
		print(pos_predator2)


		observation, reward, termination, truncation, info = env.last() #에이전트의 최신 정보


		#adfg=env.state()  # receives the entire state 특정상태에서 전체 state의 정보를 가져온다.
		print("agent:",agent)
		#print("reward",reward)

		if termination or truncation:
			print(agent, ' is terminated')
			env.step(None) # need this

			continue
		else:
			action = env.action_space(agent).sample()
			print(env.action_space(agent)) #Discrete(9)이라고 나오네.
			env.step(action)


	#env.state() # receives the entire state