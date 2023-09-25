from magent2.environments import hetero_adversarial_v1
#from magent2.environments import adversarial_pursuit_v4

render_mode = 'human'
#render_mode = 'rgb_array'
#env=env(map_size=45, minimap_mode=False, tag_penalty=-0.2,max_cycles=500, extra_features=True, render_mode=render_mode)
env = hetero_adversarial_v1.env(map_size=45, minimap_mode=False, tag_penalty=-0.2,
								max_cycles=50000, extra_features=False,render_mode=render_mode)

#env.reset()

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


		observation, reward, termination, truncation, info = env.last()
		#print("agent",observation)

		#adfg=env.state()  # receives the entire state 특정상태에서 전체 state의 정보를 가져온다.
		print("agent:",agent)
		#print("reward",reward)

		if termination or truncation:
			print(agent, ' is terminated')
			env.step(None) # need this

			continue
		else:
			action = env.action_space(agent).sample()
			env.step(action)


	#env.state() # receives the entire state