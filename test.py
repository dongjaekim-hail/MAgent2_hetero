from magent2.environments import hetero_adversarial_v1
from MADQN_junhyeon import MADQN
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
predator1_view_range = 5
predator2_view_range = 3
shared = np.zeros(entire_state)

render_mode = 'rgb_array'
env = hetero_adversarial_v1.env(map_size=45, minimap_mode=False, tag_penalty=-0.2,
								max_cycles=50000, extra_features=False,render_mode=render_mode)

shared = np.zeros(entire_state)
MADQN = MADQN(n_predator1, n_predator2, predator1_obs, predator2_obs, dim_act ,entire_state,shared)


for ep in range(1000):
	env.reset()
	print("ep:",ep)

	for agent in env.agent_iter():
		#잡단에 있는 predator들의 절대 좌표
		handles = env.env.env.env.env.get_handles()
		pos_predator1 = env.env.env.env.env.get_pos(handles[0])
		pos_predator2 = env.env.env.env.env.get_pos(handles[1])

		#에이전트 자신에게 맞는 pos 가져오는 부분
		# if agent[9] == "1":  # 1번째 predator 집단
		# 	idx = int(agent[11:])
		# 	pos = pos_predator1[idx]
		# 	view_range = predator1_view_range
		# else:
		# 	idx = int(agent[11:])
		# 	pos = pos_predator2[idx]
		# 	view_range = predator2_view_range
		#
		# print("첫번째", pos_predator1)
		# print("첫번째",pos_predator1[1])
		# print(pos_predator2)

		#매번 agent의 iteration 마다 해당 agent에 맞는 정보들을 세팅해주어야 한다.
		#이 함수를 통해 agent에 맞는 idx, adj, pos, view_range, gdqn, target_gdqn, buffer등을 설정해준다.
		#MADQN.set_agent_info(agent , pos, view_range)

		# 에이전트의 최신 정보
		observation, reward, termination, truncation, info = env.last()
		#지금 나는 에이전트 하나당 돌아가야 하는 코드이다.

		#action을 뽑는다

		#action 을 바탕으로 행동한다.


		if termination or truncation:
			print(agent, ' is terminated')
			env.step(None) # need this

			continue
		else:
			#action = MADQN.get_action(observation, MADQN.adj, mask=None)
			action = env.action_space(agent).sample()
			# print(env.action_space(agent)) #Discrete(9)이라고 나오네.
			print("이게뭔대,",env.step(action))

		#히스토리를 저장한다.

		#히스토리가 batchsize 보다 넘게 쌓였으면 업데이트를 진행한다.
	#env.state() # receives the entire state