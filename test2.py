from magent2.environments import hetero_adversarial_v1

render_mode = 'human'
# render_mode = 'rgb_array'
env = hetero_adversarial_v1.env(map_size=45, minimap_mode=False, tag_penalty=-0.2,
								max_cycles=50000, extra_features=False,render_mode=render_mode)

for ep in range(1000):
	print("ep확인",ep)
	env.reset()

	for agent in env.agent_iter():

		observation, reward, termination, truncation, info ,agent= env.last()
		print("현재 에이전트", agent)
		observation, reward, termination, truncation, info, agent = env.last()
		print("현재 에이전트", agent)
		observation, reward, termination, truncation, info, agent = env.last()
		print("현재 에이전트", agent)

		if termination or truncation:
			print(agent, ' is terminated')
			env.step(None) # need this
			continue
		else:
			action = env.action_space(agent).sample()
			env.step(action)
			observation, reward, termination, truncation, info, agent = env.last()
			print("현재 에이전트", agent)

	# env.state() # receives the entire state
	#step 을 하면 다음에이전트의 정보를 가져옴