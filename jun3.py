#우선 config에서 option을 바꿔가면서 predator 3그룹을 만들었다.

#이는 아직 건들지 않았음
# a = gw.AgentSymbol(predator_group, index="any")  #a=agent(0,-1)
#     b = gw.AgentSymbol(prey_group, index="any")      #b=agent(1,-1)   근데 이건 한번만 실행됨
#
#
#
#                                #cfg.add_reward_rule(gw.Event(agent(0,-1), "attack", agent(1,-1)), receiver=[a, b], value=[1, -1])
#     cfg.add_reward_rule(gw.Event(a, "attack", b), receiver=[a, b], value=[1, -1])

#결국 get_config 함수에서 선언한 Config의 인스턴스인 cfg가 담고있는 값들에 변화가 생겼다.
#cfg가 담고 있는 변수객체들은 다음과 같다.

# self.config_dict = {}      #이건 우선 변할 필요가 없을 것 같음. 왜냐하면
#                     cfg.set({"map_width": map_size, "map_height": map_size})
#                     cfg.set({"minimap_mode": minimap_mode})
#                     cfg.set({"embedding_size": 10})   agent들에 관한 것이 아니기 때문
# self.agent_type_dict = {}  #이거 변했고
# self.groups = []           #이거 변했고
# self.reward_rules = []     #아직         ---->뒤에 self._serialize_event_exp(config) 에서 발생하는 오류를 잡을 때 이도 반드시 해주어야

#그런데 일단 리스트로 주었기 때문에 관련해서 좀...문제가 생길 수 있다.


#그리고 당연히..
# env = magent2.GridWorld(
#             get_config(map_size, minimap_mode, **reward_args), map_size=map_size
#         )
#이 코드에 문제가 생기겠지? 역시나 생긴다.

# 그리고 각 에이전트 타입마다 얼마나 만들건지 숫자를 정해주어야 하는데 이게 generate_map() 함수로 만들어진다.
# 여기서 의문인 점은 이렇게 따로따로 만든 predator들이 서로 협력을 하냐인데...이건 환경에서 고려할 문제는 아닌 것 같다.
# GNN과 multiagent 코드를 어떻게 구성하냐의 문제인 것 같다.

# def generate_map(self):
#     env, map_size = self.env, self.map_size
#     handles = env.get_handles()
#
#     env.add_walls(method="random", n=map_size * map_size * 0.015)
#     env.add_agents(handles[0], method="random", n=map_size * map_size * 0.00625)
#     env.add_agents(handles[1], method="random", n=map_size * map_size * 0.0125)


#어쨋든 일단 get_config으로 받은것을 GridWorld 클래스의 인스터드안에서 어떻게 처리하느냐가 관건이다.

#여기까지 하고 돌려보면
# env = magent2.GridWorld(
#             get_config(map_size, minimap_mode, **reward_args), map_size=map_size
#         )
#여기서 문제가 생기고,,
#이 밑단에서 문제가 생기는  부분을 들어가보면...
#self._serialize_event_exp(config)
#이 파트에 문제 생긴다. 그래서 serialize_event_exp(config) 여기를 봐야한다.

#그래서 다음의 코드를 보게 된다.
# def collect_agent_symbol(node, config):
#     for item in node.inputs:
#         if isinstance(item, EventNode):
#             collect_agent_symbol(item, config)
#         elif isinstance(item, AgentSymbol):
#             if item not in symbol2int:
#                 symbol2int[item] = config.symbol_ct
#                 config.symbol_ct += 1
#
#
# for rule in config.reward_rules:
#     on = rule[0]
#     receiver = rule[1]
#     for symbol in receiver:
#         if symbol not in symbol2int:
#             symbol2int[symbol] = config.symbol_ct
#             config.symbol_ct += 1
#     collect_agent_symbol(on, config)                  #여기서 collect_agent_symbol이 사용되게 된다. 즉 reward_rules을 보게 될 수 밖에 없네
                                                        #reward_rules은 앞에 get_config에서 봐야

#수정한 결과 reward_rules에는 다음과 같은 list가 담긴다.
[[<magent2.gridworld.EventNode object at 0x118352280>, [<magent2.gridworld.AgentSymbol object at 0x118352100>, <magent2.gridworld.AgentSymbol object at 0x118352220>], [1, -1], False],
[<magent2.gridworld.EventNode object at 0x1183522e0>, [<magent2.gridworld.AgentSymbol object at 0x118352160>, <magent2.gridworld.AgentSymbol object at 0x118352220>], [1, -1], False],
[<magent2.gridworld.EventNode object at 0x118352340>,[<magent2.gridworld.AgentSymbol object at 0x1183521c0>, <magent2.gridworld.AgentSymbol object at 0x118352220>], [1, -1], False]]