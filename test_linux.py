from magent2.environments import adversarial_pursuit_v4

render_option = True
if render_option:
    env = adversarial_pursuit_v4.env(map_size=45, minimap_mode=False, tag_penalty=-0.2,
    max_cycles=500, extra_features=False, render_mode = 'human')
else:
    env = adversarial_pursuit_v4.env(map_size=45, minimap_mode=False, tag_penalty=-0.2,
    max_cycles=500, extra_features=False)

env.reset()

for agent in env.agent_iter():

    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        print(agent, ' is terminated')
        env.step(None)  # need this
        continue
    else:
        action = env.action_space(agent).sample()
        env.step(action)

    if render_option:
        env.render()
# env.state() # receives the entire state