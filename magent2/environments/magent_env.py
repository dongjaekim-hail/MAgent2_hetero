import numpy as np
from gymnasium.spaces import Box, Discrete
from gymnasium.utils import seeding
from pettingzoo.utils import wrappers
from pettingzoo.utils.env import ParallelEnv

from magent2 import Renderer


def make_env(raw_env):
    def env_fn(**kwargs):
        env = raw_env(**kwargs)
        env = wrappers.AssertOutOfBoundsWrapper(env)
        env = wrappers.OrderEnforcingWrapper(env)
        return env

    return env_fn


class magent_parallel_env(ParallelEnv):
    def __init__(
        self,
        env,
        active_handles,
        names,
        map_size,
        max_cycles,
        reward_range,
        minimap_mode,
        extra_features,
        render_mode=None,
    ):
        self.map_size = map_size
        self.max_cycles = max_cycles
        self.minimap_mode = minimap_mode
        self.extra_features = extra_features
        self.env = env                              #그리고 env는 다 같은 env이다. 자식클래스인 _parallel_env(in hetero_adversial)에서
                                                    #모두 같은 env인 것. 즉 env의 주소를 계속 넘겨받고 있다는 것이다.
        self.handles = active_handles               ##[c_int(0), c_int(1)]
        self._all_handles = self.env.get_handles()  #[c_int(0), c_int(1)] config로부터 온 것 같은데...
        env.reset()
        self.generate_map()                         #희한하네....이게 돌아가? 되나보네 신기하네, 앞서서 이미 _parallel_env가 작동하기 때문에
                                                    #그 안에서 이미 generate_map() 매서드가 작동하기 때문에 돌아가는게 가능한 것 같다.
        self.team_sizes = [
            env.get_num(handle) for handle in self.handles #handle에 [c_int(0), c_int(1)]가 하나씩 들어가면서 숫자를 가져온다.
        ]  # gets updated as agents die
        self.agents = [
            f"{names[j]}_{i}"
            for j in range(len(self.team_sizes))
            for i in range(self.team_sizes[j])
        ]                                           #generate_map()함수에서 각 집단의 수를 만들어줬다. 그 수대로 agents들을 생성하고 있는 것
        self.possible_agents = self.agents[:]       #['predator_0',... 'predator_24','prey_0',..., 'prey_49'] 이걸 여기서 만드네
        num_actions = [env.get_action_space(handle)[0] for handle in self.handles]   #num_actions=[13, 9]
                                                    #Predator action options: [do_nothing, move_4, tag_8] 으로 총 13가지의 옵션
                                                    #Prey action options: [do_nothing, move_8] 으로 총 9가지의 옵션

        action_spaces_list = [                      #Discrete이 정확히 무슨 함수인지는 모르겠으나, action의 개수를 나타내는 정도로만 이해하면 될 것 같다.
            Discrete(num_actions[j])                #action_spaces_list=[Discrete(13)...Discrete(13),Discrete(9)...Discrete(9)]
            for j in range(len(self.team_sizes))    #Discrete(13)이 predator수인 25개 Discrete(9)이 prey수인 50개가 리스트에 있음.
            for i in range(self.team_sizes[j])
        ]
        # may change depending on environment config? Not sure.
        team_obs_shapes = self._calc_obs_shapes() #team_obs_shapes[(10, 10, 5), (9, 9, 5)]
        state_shape = self._calc_state_shape()    #state_shape=(45, 45, 5)
        observation_space_list = [
            Box(low=0.0, high=2.0, shape=team_obs_shapes[j], dtype=np.float32)
            for j in range(len(self.team_sizes))
            for i in range(self.team_sizes[j])
        ]
                                                #observation_space_list는 총 75개의 리스트인데, 한하나가 다음과 같다.
                                                #Box(0.0, 2.0, (10, 10, 5), float32)
                                                #...(25개있음 predator의 observation_space_list 인 것 같고,
                                                #Box(0.0, 2.0, (10, 10, 5), float32)
                                                #Box(0.0, 2.0, (9, 9, 5), float32)
                                                #...(50개 있음 prey의 observation_space_list인 것 같고)
                                                #Box(0.0, 2.0, (9, 9, 5), float32)
                                                #이것들이 왜 이런 space를 가지게 되었는지는 모르겠으나 이를 이해하기 위해서는
                                                # _calc_obs_shapes()와 _calc_state_shape()를 이해해야 하는데 이 코드들 따라가보면
                                                #결국 GridWorld 클래스의 초반 부분을 이해해야 한다. 그런데 이것이 C언어로 이루어져 있는
                                                #GYM환경의 코드를 이해해야 한다. 그래서 일단은 여기까지만 이해하자.
        self.state_space = Box(low=0.0, high=2.0, shape=state_shape, dtype=np.float32)
        reward_low, reward_high = reward_range

        if extra_features:
            for space in observation_space_list:
                idx = space.shape[2] - 3 if minimap_mode else space.shape[2] - 1
                space.low[:, :, idx] = reward_low
                space.high[:, :, idx] = reward_high
            idx_state = (
                self.state_space.shape[2] - 1
                if minimap_mode
                else self.state_space.shape[2] - 1
            )
            self.state_space.low[:, :, idx_state] = reward_low
            self.state_space.high[:, :, idx_state] = reward_high

        self.action_spaces = {
            agent: space for agent, space in zip(self.agents, action_spaces_list)
        }
        self.observation_spaces = {
            agent: space for agent, space in zip(self.agents, observation_space_list)
        }

        self._zero_obs = {
            agent: np.zeros_like(space.low)
            for agent, space in self.observation_spaces.items()
        }
        self.base_state = np.zeros(self.state_space.shape, dtype="float32")
        walls = self.env._get_walls_info()
        wall_x, wall_y = zip(*walls)
        self.base_state[wall_x, wall_y, 0] = 1
        self.render_mode = render_mode
        self._renderer = None
        self.frames = 0

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def seed(self, seed=None):
        if seed is None:
            _, seed = seeding.np_random()
        self.env.set_seed(seed)

    def _calc_obs_shapes(self):
        view_spaces = [self.env.get_view_space(handle) for handle in self.handles]
        feature_spaces = [self.env.get_feature_space(handle) for handle in self.handles]
        assert all(len(tup) == 3 for tup in view_spaces)
        assert all(len(tup) == 1 for tup in feature_spaces)
        feat_size = [[fs[0]] for fs in feature_spaces]
        for feature_space in feat_size:
            if not self.extra_features:
                feature_space[0] = 2 if self.minimap_mode else 0
        obs_spaces = [           #obs_spaces=[(10, 10, 5), (9, 9, 5)]
            (view_space[:2] + (view_space[2] + feature_space[0],))
            for view_space, feature_space in zip(view_spaces, feat_size)
        ]
        return obs_spaces

    def _calc_state_shape(self):
        feature_spaces = [
            self.env.get_feature_space(handle) for handle in self._all_handles
        ]
        self._minimap_features = 2 if self.minimap_mode else 0
        # map channel and agent pair channel. Remove global agent position when minimap mode and extra features
        state_depth = (
            (max(feature_spaces)[0] - self._minimap_features) * self.extra_features
            + 1
            + len(self._all_handles) * 2
        )

        return (self.map_size, self.map_size, state_depth)

    def render(self):
        if self.render_mode is None:
            # gymnasium.logger.WARN(
            #     "You are calling render method without specifying any render mode."
            # )
            return

        if self._renderer is None:
            self._renderer = Renderer(self.env, self.map_size, self.render_mode)
        assert (
            self.render_mode == self._renderer.mode
        ), "mode must be consistent across render calls"
        return self._renderer.render(self.render_mode)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def reset(self, seed=None, return_info=False, options=None):
        if seed is not None:
            self.seed(seed=seed)
        self.agents = self.possible_agents[:]
        self.env.reset()
        self.frames = 0
        self.team_sizes = [self.env.get_num(handle) for handle in self.handles]
        self.generate_map()
        return self._compute_observations()

    def _compute_observations(self):
        observes = [None] * self.max_num_agents
        for handle in self.handles:
            ids = self.env.get_agent_id(handle)
            view, features = self.env.get_observation(handle)

            if self.minimap_mode and not self.extra_features:
                features = features[:, -2:]
            if self.minimap_mode or self.extra_features:
                feat_reshape = np.expand_dims(np.expand_dims(features, 1), 1)
                feat_img = np.tile(feat_reshape, (1, view.shape[1], view.shape[2], 1))
                fin_obs = np.concatenate([view, feat_img], axis=-1)
            else:
                fin_obs = np.copy(view)
            for id, obs in zip(ids, fin_obs):
                observes[id] = obs

        ret_agents = set(self.agents)
        return {
            agent: obs if obs is not None else self._zero_obs[agent]
            for agent, obs in zip(self.possible_agents, observes)
            if agent in ret_agents
        }

    def _compute_rewards(self):
        rewards = np.zeros(self.max_num_agents)
        for handle in self.handles:
            ids = self.env.get_agent_id(handle)
            rewards[ids] = self.env.get_reward(handle)
        ret_agents = set(self.agents)
        return {
            agent: float(rew)
            for agent, rew in zip(self.possible_agents, rewards)
            if agent in ret_agents
        }

    def _compute_terminates(self, step_done):
        dones = np.ones(self.max_num_agents, dtype=bool)
        if not step_done:
            for i, handle in enumerate(self.handles):
                ids = self.env.get_agent_id(handle)
                dones[ids] = ~self.env.get_alive(handle)
                self.team_sizes[i] = len(ids) - np.array(dones[ids]).sum()
        ret_agents = set(self.agents)
        return {
            agent: bool(done)
            for agent, done in zip(self.possible_agents, dones)
            if agent in ret_agents
        }

    def state(self):
        """Returns an observation of the global environment."""
        state = np.copy(self.base_state)

        for handle in self._all_handles:
            view, features = self.env.get_observation(handle)

            pos = self.env.get_pos(handle)
            pos_x, pos_y = zip(*pos)
            state[pos_x, pos_y, 1 + handle.value * 2] = 1
            state[pos_x, pos_y, 2 + handle.value * 2] = view[
                :, view.shape[1] // 2, view.shape[2] // 2, 2
            ]

            if self.extra_features:
                add_zeros = np.zeros(
                    (
                        features.shape[0],
                        state.shape[2]
                        - (
                            1
                            + len(self.team_sizes) * 2
                            + features.shape[1]
                            - self._minimap_features
                        ),
                    )
                )

                rewards = features[:, -1 - self._minimap_features]
                actions = features[:, : -1 - self._minimap_features]
                actions = np.concatenate((actions, add_zeros), axis=1)
                rewards = rewards.reshape(len(rewards), 1)
                state_features = np.hstack((actions, rewards))

                state[pos_x, pos_y, 1 + len(self.team_sizes) * 2 :] = state_features
        return state

    def step(self, all_actions):
        action_list = [-1] * len(self.agents)
        for i, agent in enumerate(self.agents):
            if agent in all_actions:
                action_list[i] = all_actions[agent]

        all_actions = np.asarray(action_list, dtype=np.int32)
        start_point = 0
        for i in range(len(self.handles)):
            size = self.team_sizes[i]
            self.env.set_action(
                self.handles[i], all_actions[start_point : (start_point + size)]
            )
            start_point += size

        self.frames += 1

        step_done = self.env.step()

        truncations = {agent: self.frames >= self.max_cycles for agent in self.agents}
        terminations = self._compute_terminates(step_done)
        observations = self._compute_observations()
        rewards = self._compute_rewards()
        infos = {agent: {} for agent in self.agents}
        self.env.clear_dead()
        self.agents = [
            agent
            for agent in self.agents
            if not (terminations[agent] or truncations[agent])
        ]

        if self.render_mode == "human":
            self.render()
        return observations, rewards, terminations, truncations, infos
