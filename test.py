from magent2.environments import hetero_adversarial_v1

render_mode = 'human'
# render_mode = 'rgb_array'
env = adversarial_pursuit_v4.env(map_size=45, minimap_mode=False, tag_penalty=-0.2,
max_cycles=500, extra_features=False,render_mode=render_mode)

for ep in range(1000):
	env.reset()

	for agent in env.agent_iter():

		observation, reward, termination, truncation, info = env.last()

		if termination or truncation:
			print(agent, ' is terminated')
			env.step(None) # need this
			continue
		else:
			action = env.action_space(agent).sample()
			env.step(action)

	# env.state() # receives the entire state