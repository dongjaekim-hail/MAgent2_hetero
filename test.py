from magent2.environments import hetero_adversarial_v1

render_mode = 'human'
# render_mode = 'rgb_array'
env = hetero_adversarial_v1.env(map_size=20, minimap_mode=False, tag_penalty=-0.2,
max_cycles=50000, extra_features=False,render_mode=render_mode)

#env.reset()

for ep in range(1000):
	env.reset()
	print("ep:",ep)

	for agent in env.agent_iter():

		observation, reward, termination, truncation, info = env.last()
		#env.state()  # receives the entire state 특정상태에서 전체 state의 정보를 가져온다.
		if termination or truncation:
			print(agent, ' is terminated')
			env.step(None) # need this
			continue
		else:
			action = env.action_space(agent).sample()
			env.step(action)

	#env.state() # receives the entire state