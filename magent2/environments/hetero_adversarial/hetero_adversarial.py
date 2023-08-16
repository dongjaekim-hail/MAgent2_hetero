# noqa
"""
# Adversarial Pursuit

```{figure} magent_adversarial_pursuit.gif
:width: 140px
:name: adversarial_pursuit
```

This environment is part of the <a href='..'>MAgent2 environments</a>. Please read that page first for general information.

| Import             | `from magent2.environments import adversarial_pursuit_v4` |
|--------------------|--------------------------------------------------------|
| Actions            | Discrete                                               |
| Parallel API       | Yes                                                    |
| Manual Control     | No                                                     |
| Agents             | `agents= [predator_[0-24], prey_[0-49]]`               |
| Agents             | 75                                                     |
| Action Shape       | (9),(13)                                               |
| Action Values      | Discrete(9),(13)                                       |
| Observation Shape  | (9,9,5), (10,10,9)                                     |
| Observation Values | [0,2]                                                  |
| State Shape        | (45, 45, 5)                                            |
| State Values       | (0, 2)                                                 |

```{figure} ../../_static/img/aec/magent_adversarial_pursuit_aec.svg
:width: 200px
:name: adversarial_pursuit
```

The red agents must navigate the obstacles and tag (similar to attacking, but without damaging) the blue agents. The blue agents should try to avoid being tagged. To be effective, the red agents, who are much are slower and larger than the blue agents, must work together to trap blue agents so
they can be tagged continually.

### Arguments

``` python
adversarial_pursuit_v4.env(map_size=45, minimap_mode=False, tag_penalty=-0.2,
max_cycles=500, extra_features=False)
```

`map_size`: Sets dimensions of the (square) map. Increasing the size increases the number of agents. Minimum size is 7.

`minimap_mode`: Turns on global minimap observations. These observations include your and your opponents piece densities binned over the 2d grid of the observation space. Also includes your `agent_position`, the absolute position on the map (rescaled from 0 to 1).

`tag_penalty`:  reward when red agents tag anything

`max_cycles`:  number of frames (a step for each agent) until game terminates

`extra_features`: Adds additional features to observation (see table). Default False

#### Action Space

Key: `move_N` means N separate actions, one to move to each of the N nearest squares on the grid.

Predator action options: `[do_nothing, move_4, tag_8]`

Prey action options: `[do_nothing, move_8]`

#### Reward

Predator's reward is given as:

* 1 reward for tagging a prey
* -0.2 reward for tagging anywhere (`tag_penalty` option)

Prey's reward is given as:

* -1 reward for being tagged


#### Observation space

The observation space is a 10x10 map for pursuers and a 9x9 map for the pursued. They contain the following channels, which are (in order):

feature | number of channels
--- | ---
obstacle/off the map| 1
my_team_presence| 1
my_team_hp| 1
other_team_presence| 1
other_team_hp| 1
binary_agent_id(extra_features=True)| 10
one_hot_action(extra_features=True)| 9/Prey,13/Predator
last_reward(extra_features=True)| 1

### State space

The observation space is a 45x45 map. It contains the following channels, which are (in order):

feature | number of channels
--- | ---
obstacle map| 1
prey_presence| 1
prey_hp| 1
predator_presence| 1
predator_hp| 1
binary_agent_id(extra_features=True)| 10
one_hot_action(extra_features=True)|  13 (max action space)
last_reward(extra_features=True)| 1


### Version History

* v4: Underlying library fix (1.18.0)
* v3: Fixed bugs and changed default parameters (1.7.0)
* v2: Observation space bound fix, bumped version of all environments due to adoption of new agent iteration scheme where all agents are iterated over after they are done (1.4.0)
* v1: Agent order under death changed (1.3.0)
* v0: Initial versions release (1.0.0)

"""

import numpy as np
from gymnasium.utils import EzPickle
from pettingzoo.utils.conversions import parallel_to_aec_wrapper

import magent2
from magent2.environments.magent_env import magent_parallel_env, make_env

default_map_size = 45
max_cycles_default = 500
minimap_mode_default = False
default_reward_args = dict(tag_penalty=-0.2)

# env = hetero_adversarial_v1.env(map_size=45, minimap_mode=False, tag_penalty=-0.2,
# max_cycles=500, extra_features=False,render_mode=render_mode)
# 이 인자들이 raw_env에 가게 된다.
# tag_penalty=-0.2 와 render_mode=render_mode가 reward_args에 들어가 딕셔너리의 형태로 저장된다.
# 결국 tag_penalty=-0.2 만 reward_args에 딕셔너리 형태로 저장된다.

def parallel_env(
    map_size=default_map_size,
    max_cycles=max_cycles_default,
    minimap_mode=minimap_mode_default,
    extra_features=False,
    render_mode=None,
    **reward_args
):
    env_reward_args = dict(**default_reward_args)
    env_reward_args.update(reward_args)
    return _parallel_env(
        map_size, minimap_mode, env_reward_args, max_cycles, extra_features, render_mode
    )

def raw_env(
    map_size=default_map_size,
    max_cycles=max_cycles_default,
    minimap_mode=minimap_mode_default,
    extra_features=False,
    **reward_args
):
    return parallel_to_aec_wrapper(
        parallel_env(map_size, max_cycles, minimap_mode, extra_features, **reward_args)
    )


env = make_env(raw_env)

# get_config(map_size, minimap_mode, **reward_args)이고, reward_args에는
# {’tag_penalty’ : -0.2}가 넘어간다.
# get_config함수는 한번만 실행되므로...안에서 for문을 이용해야지 여러개의 predator들을 만들 수 있을 것 같다.
def get_config(map_size, minimap_mode, tag_penalty):
    gw = magent2.gridworld
    cfg = gw.Config()
                                    #그냥 우선 gridworld 통채로 가져오고 그 중에서 Config 클래스 객체만 하나 만든다.
                                    #이때, cfg에는 다음과 같은  변수들이 존재한다.
                                    # self.config_dict = {}
                                    # self.agent_type_dict = {}
                                    # self.groups = []
                                    # self.reward_rules = []


                                    #cfg.set( )에 넘겨준 인자들(딕셔너리들)을 config_dict에 그대로 복사한다.
    cfg.set({"map_width": map_size, "map_height": map_size})
    cfg.set({"minimap_mode": minimap_mode})
    cfg.set({"embedding_size": 10})


    options = {
        "width": 2,
        "length": 2,
        "hp": 1,
        "speed": 1,
        "view_range": gw.CircleRange(5),
        "attack_range": gw.CircleRange(2),
        "attack_penalty": tag_penalty,
    }
    predator = cfg.register_agent_type("predator", options)
                                    #self.agent_type_dict에 다음과 같이 predator의 option을 딕셔너리형태로 저장함.
                                    #{'predator': {'attack_penalty': -0.2, 'attack_range': circle(2), 'hp': 1,
                                    # 'length': 2, 'speed': 1, 'view_range': circle(5), 'width': 2}}

    options = {
        "width": 1,
        "length": 1,
        "hp": 1,
        "speed": 1.5,
        "view_range": gw.CircleRange(4),    #아 이거 객체구나
        "attack_range": gw.CircleRange(0),
    }
    prey = cfg.register_agent_type("prey", options)
    #결국,,,,agent_type_dict에 다음과 같이 저장이 된다.
    #{'predator': {'attack_penalty': -0.2, 'attack_range': circle(2), 'hp': 1,'length': 2, 'speed': 1, 'view_range': circle(5), 'width': 2},
    # 'prey':{'attack_range': circle(0), 'hp': 1, 'length': 1, 'speed': 1.5, 'view_range': circle(4), 'width': 1}}


    predator_group = cfg.add_group(predator)         #cfg의 groups 리스트에 predator를 저장하고, 0을 반환한다. predator_group=0
    prey_group = cfg.add_group(prey)                 #cfg의 groups 리스트에 prey를 저장하고, 1을 반환한다. prey_group=1


                                                     #a = gw.AgentSymbol(0, index="any")을 넣는 것.
    a = gw.AgentSymbol(predator_group, index="any")  #a=agent(0,-1)
    b = gw.AgentSymbol(prey_group, index="any")      #b=agent(1,-1)   근데 이건 한번만 실행됨



                               #cfg.add_reward_rule(gw.Event(agent(0,-1), "attack", agent(1,-1)), receiver=[a, b], value=[1, -1])
    cfg.add_reward_rule(gw.Event(a, "attack", b), receiver=[a, b], value=[1, -1])
                                #gw.Event를 통해 gridworld에서 만든 Eventnode클래스의 객체인 Event 객체에 값을 주어 __call__을 호출한다.
                                #__call__내부에는 node=Evnetnode()객체를 호출한다. 따라서 node객체에는
                                # node = EventNode()
                                # node.op = EventNode.OP_NOT     #node.op=7
                                # node.inputs = [self]           #node.inputs=["attack",agent(1,-1)]
                                #node.predicate = predicate      #node.predicate="attack"
                                #다음과 같은 요소들이 있다. 이 요소들을 (a,"attack',b)으로 채우고, 이 node를 반환한다.
                                # Predator’s reward is given as:
                                # 1 reward for tagging a prey
                                # -0.2 reward for tagging anywhere(tag_penalty option) 아무곳이나 터치하지 못하도록 벌을 줌
                                # Prey’s reward is given as:
                                # -1 reward for being tagged
                                #attack이라는 행위에 대해서 a(predator)은 1점을 받고, b(prey)는 -1점을 받는다.
                                #config클래스 안에 있는 함수임. self.reward_rules = [] 과 관련있음
                                #[[<magent2.gridworld.EventNode object at 0x107745370>, [agent(0,-1), agent(1,-1)], [1, -1], False]]
                                #이와 같이, reward.rules 라는 리스트에 다음과 같이 저장한다 gw.Event는 node를 반환하기에 저렇게 객체의 주소를 담는 것을 확인할 수 있다.
    return cfg

                                #return _parallel_env(
                                #map_size, minimap_mode, env_reward_args, max_cycles, extra_features, render_mode
class _parallel_env(magent_parallel_env, EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "hetero_adversarial_v1",
        "render_fps": 10,
    }

    # default_map_size = 45,,
    # minimap_mode_default = False
    # max_cycles_default = 500
    # default_reward_args = dict(tag_penalty=-0.2)
    # extra_features=False
    def __init__(
        self,
        map_size,
        minimap_mode,
        reward_args,
        max_cycles,
        extra_features,
        render_mode=None,
    ):
        EzPickle.__init__(
            self,
            map_size,
            minimap_mode,
            reward_args,
            max_cycles,
            extra_features,
            render_mode,
        )
        assert map_size >= 7, "size of map must be at least 7"
        env = magent2.GridWorld(
            get_config(map_size, minimap_mode, **reward_args), map_size=map_size
        )
        #get_config의 결과로 gridworld.Config  클래스의 인스턴스를 반환한다. 이 인스턴스의 agent_type_dict인자에 range_view가 들어간다.
        #따라서, GridWorld 안에서 agent_type_dict안에 있는 range_view가 어떻게 돌아다니는지 살펴보아야 한다.
        #class GridWorld
        #def __init__(self, config, **kwargs): magent2.GridWorld의 인자이다. config이외의 키워드 인자는 딕셔너리로 흡수한다.


        handles = env.get_handles()                                                 #[c_int(0), c_int(1)]
        reward_vals = np.array([1, -1, -1, -1, -1] + list(reward_args.values()))    #[ 1.  -1.  -1.  -1.  -1.  -0.2]
        reward_range = [                                                            #[-4.2, 1.0]
            np.minimum(reward_vals, 0).sum(),
            np.maximum(reward_vals, 0).sum(),
        ]
        names = ["predator", "prey"]
                             #중요한 건 이 파트구나.
        super().__init__(    #magent_parallel_env 클래스의 __init__을 실행하는 것!
            env,                #env = magent2.GridWorld(get_config(map_size, minimap_mode, **reward_args), map_size=map_size)
            handles,            #[c_int(0), c_int(1)]
            names,              #names = ["predator", "prey"]
            map_size,           #map_size=45
            max_cycles,         #max_cycles=500
            reward_range,       #[-4.2, 1.0]
            minimap_mode,       #minimap_mode=False
            extra_features,     #extra_features=False
            render_mode,        #render_mode=None
        )

    def generate_map(self):
        env, map_size = self.env, self.map_size
        handles = env.get_handles()

        env.add_walls(method="random", n=map_size * map_size * 0.015)
        env.add_agents(handles[0], method="random", n=map_size * map_size * 0.00625)
        env.add_agents(handles[1], method="random", n=map_size * map_size * 0.0125)
