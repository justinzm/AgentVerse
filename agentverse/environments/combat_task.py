import asyncio
import logging
from typing import Any, Dict, List
from collections import defaultdict

# from agentverse.agents.agent import Agent
import numpy as np

from agentverse.agents.conversation_agent import BaseAgent
from agentverse.environments.rules.base import Rule
from agentverse.message import Message

from . import env_registry as EnvironmentRegistry
from .base import BaseEnvironment


@EnvironmentRegistry.register("combat")
class CombatEnvironment(BaseEnvironment):
    class Config:
        arbitrary_types_allowed = True
    """
    A basic environment implementing the logic of conversation.

    Args:
        agents: List of agents
        rule: Rule for the environment
        max_turns: Maximum number of turns
        cnt_turn: Current turn number
        last_messages: Messages from last turn
        rule_params: Variables set by the rule
    """

    rule: Rule
    agents: List[BaseAgent]
    # name of enemys
    enemys: List[str]
    # key: name of enemy, value: coordinate
    enemys_coordinate: dict = {}
    # key: name of enemy, value: health point
    enemys_healthpoint: defaultdict = defaultdict(int)
    # key: name of agent, value: agent survives or not
    agents_survive: defaultdict = defaultdict(int)
    # key: name of enemy, value: enemy survives or not 
    enemys_survive: defaultdict = defaultdict(int)
    agents_num: int = 5
    enemys_num: int = 5
    max_turns: int = 10
    cnt_turn: int = 0
    grids: np.ndarray = None
    # 17 = 1 + 15 + 1 (including padding * 2)
    grids_dim: int = 17
    # key: coordinate, value: name of bot
    grids_occupancy: defaultdict = defaultdict(str)
    last_messages: List[Message] = []
    rule_params: Dict = {}

    def __init__(self, rule, **kwargs):
        # set the enemys
        tmp_enemys = ["enemy_1", "enemy_2", "enemy_3", "enemy_4", "enemy_5"]

        # set coordinates of enemys
        tmp_enemys_coordinate = {}
        tmp_enemys_coordinate["enemy_1"] = [2,10]
        tmp_enemys_coordinate["enemy_2"] = [5,13]
        tmp_enemys_coordinate["enemy_3"] = [8,10]
        tmp_enemys_coordinate["enemy_4"] = [11,13]
        tmp_enemys_coordinate["enemy_5"] = [14,10]

        # set health point of enemys
        tmp_enemys_healthpoint = {}
        tmp_enemys_healthpoint["enemy_1"] = 3
        tmp_enemys_healthpoint["enemy_2"] = 3
        tmp_enemys_healthpoint["enemy_3"] = 3
        tmp_enemys_healthpoint["enemy_4"] = 3
        tmp_enemys_healthpoint["enemy_5"] = 3

        # all agents survive at first
        tmp_agents_survive = {}
        tmp_agents_survive["bot_1"] = 1
        tmp_agents_survive["bot_2"] = 1
        tmp_agents_survive["bot_3"] = 1
        tmp_agents_survive["bot_4"] = 1
        tmp_agents_survive["bot_5"] = 1

        # all enemys survive at first
        tmp_enemys_survive = {}
        tmp_enemys_survive["enemy_1"] = 1
        tmp_enemys_survive["enemy_2"] = 1
        tmp_enemys_survive["enemy_3"] = 1
        tmp_enemys_survive["enemy_4"] = 1
        tmp_enemys_survive["enemy_5"] = 1

        # set grids occupancy
        tmp_grids_occupancy = {}
        tmp_grids_occupancy["2_10"] = "enemy_1"
        tmp_grids_occupancy["5_13"] = "enemy_2"
        tmp_grids_occupancy["8_10"] = "enemy_3"
        tmp_grids_occupancy["11_13"] = "enemy_4"
        tmp_grids_occupancy["14_10"] = "enemy_5"
        tmp_grids_occupancy["2_3"] = "bot_1"
        tmp_grids_occupancy["5_6"] = "bot_2"
        tmp_grids_occupancy["8_3"] = "bot_3"
        tmp_grids_occupancy["11_6"] = "bot_4"
        tmp_grids_occupancy["14_3"] = "bot_5"

        # set the rule
        rule_config = rule
        order_config = rule_config.get("order", {"type": "sequential"})
        visibility_config = rule_config.get("visibility", {"type": "all"})
        selector_config = rule_config.get("selector", {"type": "basic"})
        updater_config = rule_config.get("updater", {"type": "basic"})
        describer_config = rule_config.get("describer", {"type": "basic"})
        rule = Rule(
            order_config,
            visibility_config,
            selector_config,
            updater_config,
            describer_config,
        )
        
        # init
        super().__init__(
            rule=rule, 
            enemys = tmp_enemys,
            enemys_coordinate = tmp_enemys_coordinate,
            enemys_healthpoint = tmp_enemys_healthpoint,
            agents_survive = tmp_agents_survive,
            enemys_survive = tmp_enemys_survive,
            grids_occupancy = tmp_grids_occupancy,
            **kwargs
            )

    async def step(self) -> List[Message]:
        """Run one step of the environment"""

        # update the enemy
        if (self.cnt_turn % 10 == 0):
            # enemy_1 move(up)
            if self.enemys_survive["enemy_1"] == 1:
                if f"{self.enemys_coordinate[self.enemys[0]][0] - 1}_{self.enemys_coordinate[self.enemys[0]][1]}" in self.grids_occupancy:
                    if self.grids_occupancy[f"{self.enemys_coordinate[self.enemys[0]][0] - 1}_{self.enemys_coordinate[self.enemys[0]][1]}"] == "nobody":
                        self.grids_occupancy[f"{self.enemys_coordinate[self.enemys[0]][0]}_{self.enemys_coordinate[self.enemys[0]][1]}"] = "nobody"
                        self.grids_occupancy[f"{self.enemys_coordinate[self.enemys[0]][0] - 1}_{self.enemys_coordinate[self.enemys[0]][1]}"] = "enemy_1"
                        self.enemys_coordinate[self.enemys[0]][0] -= 1
                        logging.info("enemy_1 move(up)")
                    else:
                        logging.info("enemy_1 stay()")
                else:
                    self.grids_occupancy[f"{self.enemys_coordinate[self.enemys[0]][0]}_{self.enemys_coordinate[self.enemys[0]][1]}"] = "nobody"
                    self.grids_occupancy[f"{self.enemys_coordinate[self.enemys[0]][0] - 1}_{self.enemys_coordinate[self.enemys[0]][1]}"] = "enemy_1"
                    self.enemys_coordinate[self.enemys[0]][0] -= 1
                    logging.info("enemy_1 move(up)")
        elif (self.cnt_turn % 10 == 1):
            # enemy_2 attack(bot_{target + 1})
            if self.enemys_survive["enemy_2"] == 1:
                target = 0
                inRange2 = False
                for x in [self.enemys_coordinate["enemy_2"][0] - 1, self.enemys_coordinate["enemy_2"][0], self.enemys_coordinate["enemy_2"][0] + 1]:
                    if inRange2:
                        break
                    for y in [self.enemys_coordinate["enemy_2"][1] - 1, self.enemys_coordinate["enemy_2"][1], self.enemys_coordinate["enemy_2"][1] + 1]:
                        if inRange2:
                            break
                        if f"{x}_{y}" in self.grids_occupancy: 
                            if self.grids_occupancy[f"{x}_{y}"] != "nobody":
                                for i in range(len(self.agents)):
                                    if self.grids_occupancy[f"{x}_{y}"] == self.agents[i].name:
                                        target = i
                                        inRange2 = True
                                        self.agents[target].health_point -= 1
                                        logging.info(f"enemy_2 attack(bot_{target + 1})")
                                        # about die
                                        if self.agents[target].health_point == 0:
                                            logging.info(f"{self.agents[target].name} is dead!")
                                            self.agents_num -= 1
                                            self.agents_survive[self.agents[target].name] = 0
                                            self.grids_occupancy[f"{x}_{y}"] = "nobody"
                                            for agent in self.agents:
                                                agent.others_coordinates[self.agents[target].name] = "Dead"
                                        break
                if not inRange2:
                    logging.info("enemy_2 stay()")
        elif (self.cnt_turn % 10 == 2):
            # enemy_3 move(down)
            if self.enemys_survive["enemy_3"] == 1:
                if f"{self.enemys_coordinate[self.enemys[2]][0] + 1}_{self.enemys_coordinate[self.enemys[2]][1]}" in self.grids_occupancy:
                    if self.grids_occupancy[f"{self.enemys_coordinate[self.enemys[2]][0] + 1}_{self.enemys_coordinate[self.enemys[2]][1]}"] == "nobody":
                        self.grids_occupancy[f"{self.enemys_coordinate[self.enemys[2]][0]}_{self.enemys_coordinate[self.enemys[2]][1]}"] = "nobody"
                        self.grids_occupancy[f"{self.enemys_coordinate[self.enemys[2]][0] + 1}_{self.enemys_coordinate[self.enemys[2]][1]}"] = "enemy_3"
                        self.enemys_coordinate[self.enemys[2]][0] += 1
                        logging.info("enemy_3 move(down)")
                    else:
                        logging.info("enemy_3 stay()")
                else:
                    self.grids_occupancy[f"{self.enemys_coordinate[self.enemys[2]][0]}_{self.enemys_coordinate[self.enemys[2]][1]}"] = "nobody"
                    self.grids_occupancy[f"{self.enemys_coordinate[self.enemys[2]][0] + 1}_{self.enemys_coordinate[self.enemys[2]][1]}"] = "enemy_3"
                    self.enemys_coordinate[self.enemys[2]][0] += 1
                    logging.info("enemy_3 move(down)")
        elif (self.cnt_turn % 10 == 3):
            # enemy_4 attack(bot_{target + 1})
            if self.enemys_survive["enemy_4"] == 1:
                target = 0
                inRange4 = False
                for x in [self.enemys_coordinate["enemy_4"][0] - 1, self.enemys_coordinate["enemy_4"][0], self.enemys_coordinate["enemy_4"][0] + 1]:
                    if inRange4:
                        break
                    for y in [self.enemys_coordinate["enemy_4"][1] - 1, self.enemys_coordinate["enemy_4"][1], self.enemys_coordinate["enemy_4"][1] + 1]:
                        if inRange4:
                            break
                        if f"{x}_{y}" in self.grids_occupancy: 
                            if self.grids_occupancy[f"{x}_{y}"] != "nobody":
                                for i in range(len(self.agents)):
                                    if self.grids_occupancy[f"{x}_{y}"] == self.agents[i].name:
                                        target = i
                                        inRange4 = True
                                        self.agents[target].health_point -= 1
                                        logging.info(f"enemy_4 attack(bot_{target + 1})")
                                        # about die
                                        if self.agents[target].health_point == 0:
                                            logging.info(f"{self.agents[target].name} is dead!")
                                            self.agents_num -= 1
                                            self.agents_survive[self.agents[target].name] = 0
                                            self.grids_occupancy[f"{x}_{y}"] = "nobody"
                                            for agent in self.agents:
                                                agent.others_coordinates[self.agents[target].name] = "Dead"
                                        break
                if not inRange4:
                    logging.info("enemy_4 stay()")
        elif (self.cnt_turn % 10 == 4):
            # enemy_5 move(left)
            if self.enemys_survive["enemy_5"] == 1:
                if f"{self.enemys_coordinate[self.enemys[4]][0]}_{self.enemys_coordinate[self.enemys[4]][1] - 1}" in self.grids_occupancy:
                    if self.grids_occupancy[f"{self.enemys_coordinate[self.enemys[4]][0]}_{self.enemys_coordinate[self.enemys[4]][1] - 1}"] == "nobody":
                        self.grids_occupancy[f"{self.enemys_coordinate[self.enemys[4]][0]}_{self.enemys_coordinate[self.enemys[4]][1]}"] = "nobody"
                        self.grids_occupancy[f"{self.enemys_coordinate[self.enemys[4]][0]}_{self.enemys_coordinate[self.enemys[4]][1] - 1}"] = "enemy_5"
                        self.enemys_coordinate[self.enemys[4]][1] -= 1
                        logging.info("enemy_5 move(left)")
                    else:
                        logging.info("enemy_5 stay()")
                else:
                    self.grids_occupancy[f"{self.enemys_coordinate[self.enemys[4]][0]}_{self.enemys_coordinate[self.enemys[4]][1]}"] = "nobody"
                    self.grids_occupancy[f"{self.enemys_coordinate[self.enemys[4]][0]}_{self.enemys_coordinate[self.enemys[4]][1] - 1}"] = "enemy_5"
                    self.enemys_coordinate[self.enemys[4]][1] -= 1
                    logging.info("enemy_5 move(left)")
        elif (self.cnt_turn % 10 == 5):
            # enemy_1 attack(bot_{target + 1})
            if self.enemys_survive["enemy_1"] == 1:
                target = 0
                inRange1 = False
                for x in [self.enemys_coordinate["enemy_1"][0] - 1, self.enemys_coordinate["enemy_1"][0], self.enemys_coordinate["enemy_1"][0] + 1]:
                    if inRange1:
                        break
                    for y in [self.enemys_coordinate["enemy_1"][1] - 1, self.enemys_coordinate["enemy_1"][1], self.enemys_coordinate["enemy_1"][1] + 1]:
                        if inRange1:
                            break
                        if f"{x}_{y}" in self.grids_occupancy: 
                            if self.grids_occupancy[f"{x}_{y}"] != "nobody":
                                for i in range(len(self.agents)):
                                    if self.grids_occupancy[f"{x}_{y}"] == self.agents[i].name:
                                        target = i
                                        inRange1 = True
                                        self.agents[target].health_point -= 1
                                        logging.info(f"enemy_1 attack(bot_{target + 1})")
                                        # about die
                                        if self.agents[target].health_point == 0:
                                            logging.info(f"{self.agents[target].name} is dead!")
                                            self.agents_num -= 1
                                            self.agents_survive[self.agents[target].name] = 0
                                            self.grids_occupancy[f"{x}_{y}"] = "nobody"
                                            for agent in self.agents:
                                                agent.others_coordinates[self.agents[target].name] = "Dead"
                                        break
                if not inRange1:
                    logging.info("enemy_1 stay()")
        elif (self.cnt_turn % 10 == 6):    
            # enemy_2 move(right)
            if self.enemys_survive["enemy_2"] == 1:
                if f"{self.enemys_coordinate[self.enemys[1]][0]}_{self.enemys_coordinate[self.enemys[1]][1] + 1}" in self.grids_occupancy:
                    if self.grids_occupancy[f"{self.enemys_coordinate[self.enemys[1]][0]}_{self.enemys_coordinate[self.enemys[1]][1] + 1}"] == "nobody":
                        self.grids_occupancy[f"{self.enemys_coordinate[self.enemys[1]][0]}_{self.enemys_coordinate[self.enemys[1]][1]}"] = "nobody"
                        self.grids_occupancy[f"{self.enemys_coordinate[self.enemys[1]][0]}_{self.enemys_coordinate[self.enemys[1]][1] + 1}"] = "enemy_2"
                        self.enemys_coordinate[self.enemys[1]][1] += 1
                        logging.info("enemy_2 move(right)")
                    else:
                        logging.info("enemy_2 stay()")
                else:
                    self.grids_occupancy[f"{self.enemys_coordinate[self.enemys[1]][0]}_{self.enemys_coordinate[self.enemys[1]][1]}"] = "nobody"
                    self.grids_occupancy[f"{self.enemys_coordinate[self.enemys[1]][0]}_{self.enemys_coordinate[self.enemys[1]][1] + 1}"] = "enemy_2"
                    self.enemys_coordinate[self.enemys[1]][1] += 1
                    logging.info("enemy_2 move(right)")
        elif (self.cnt_turn % 10 == 7):
            # enemy_3 attack(bot_{target + 1})
            if self.enemys_survive["enemy_3"] == 1:
                target = 0
                inRange3 = False
                for x in [self.enemys_coordinate["enemy_3"][0] - 1, self.enemys_coordinate["enemy_3"][0], self.enemys_coordinate["enemy_3"][0] + 1]:
                    if inRange3:
                        break
                    for y in [self.enemys_coordinate["enemy_3"][1] - 1, self.enemys_coordinate["enemy_3"][1], self.enemys_coordinate["enemy_3"][1] + 1]:
                        if inRange3:
                            break
                        if f"{x}_{y}" in self.grids_occupancy: 
                            if self.grids_occupancy[f"{x}_{y}"] != "nobody":
                                for i in range(len(self.agents)):
                                    if self.grids_occupancy[f"{x}_{y}"] == self.agents[i].name:
                                        target = i
                                        inRange3 = True
                                        self.agents[target].health_point -= 1
                                        logging.info(f"enemy_3 attack(bot_{target + 1})")
                                        # about die
                                        if self.agents[target].health_point == 0:
                                            logging.info(f"{self.agents[target].name} is dead!")
                                            self.agents_num -= 1
                                            self.agents_survive[self.agents[target].name] = 0
                                            self.grids_occupancy[f"{x}_{y}"] = "nobody"
                                            for agent in self.agents:
                                                agent.others_coordinates[self.agents[target].name] = "Dead"
                                        break
                if not inRange3:
                    logging.info("enemy_3 stay()")
        elif (self.cnt_turn % 10 == 8):
            # enemy_4 move(up)
            if self.enemys_survive["enemy_4"] == 1:
                if f"{self.enemys_coordinate[self.enemys[3]][0] - 1}_{self.enemys_coordinate[self.enemys[3]][1]}" in self.grids_occupancy:
                    if self.grids_occupancy[f"{self.enemys_coordinate[self.enemys[3]][0] - 1}_{self.enemys_coordinate[self.enemys[3]][1]}"] == "nobody":
                        self.grids_occupancy[f"{self.enemys_coordinate[self.enemys[3]][0]}_{self.enemys_coordinate[self.enemys[3]][1]}"] = "nobody"
                        self.grids_occupancy[f"{self.enemys_coordinate[self.enemys[3]][0] - 1}_{self.enemys_coordinate[self.enemys[3]][1]}"] = "enemy_4"
                        self.enemys_coordinate[self.enemys[3]][0] -= 1
                        logging.info("enemy_4 move(up)")
                    else:
                        logging.info("enemy_4 stay()")
                else:
                    self.grids_occupancy[f"{self.enemys_coordinate[self.enemys[3]][0]}_{self.enemys_coordinate[self.enemys[3]][1]}"] = "nobody"
                    self.grids_occupancy[f"{self.enemys_coordinate[self.enemys[3]][0] - 1}_{self.enemys_coordinate[self.enemys[3]][1]}"] = "enemy_4"
                    self.enemys_coordinate[self.enemys[3]][0] -= 1
                    logging.info("enemy_4 move(up)")
        elif (self.cnt_turn % 10 == 9):
            # enemy_5 attack(bot_{target + 1})
            if self.enemys_survive["enemy_5"] == 1:
                target = 0
                inRange5 = False
                for x in [self.enemys_coordinate["enemy_5"][0] - 1, self.enemys_coordinate["enemy_5"][0], self.enemys_coordinate["enemy_5"][0] + 1]:
                    if inRange5:
                        break
                    for y in [self.enemys_coordinate["enemy_5"][1] - 1, self.enemys_coordinate["enemy_5"][1], self.enemys_coordinate["enemy_5"][1] + 1]:
                        if inRange5:
                            break
                        if f"{x}_{y}" in self.grids_occupancy: 
                            if self.grids_occupancy[f"{x}_{y}"] != "nobody":
                                for i in range(len(self.agents)):
                                    if self.grids_occupancy[f"{x}_{y}"] == self.agents[i].name:
                                        target = i
                                        inRange5 = True
                                        self.agents[target].health_point -= 1
                                        logging.info(f"enemy_5 attack(bot_{target + 1})")
                                        # about die
                                        if self.agents[target].health_point == 0:
                                            logging.info(f"{self.agents[target].name} is dead!")
                                            self.agents_num -= 1
                                            self.agents_survive[self.agents[target].name] = 0
                                            self.grids_occupancy[f"{x}_{y}"] = "nobody"
                                            for agent in self.agents:
                                                agent.others_coordinates[self.agents[target].name] = "Dead"
                                        break
                if not inRange5:
                    logging.info("enemy_5 stay()")

        # update enemys coordinates in sight of agents
        for agent in self.agents:
            for enemy in self.enemys:
                agent.enemys_coordinates[enemy] = self.enemys_coordinate[enemy]

        # Get the next agent index
        agent_ids = []
        pre_agent_ids = self.rule.get_next_agent_idx(self)
        for id in pre_agent_ids:
            if self.agents_survive[self.agents[id].name] == 1:
                agent_ids.append(id)

        # Generate current environment description
        env_descriptions = self.rule.get_env_description(self)

        # Update the set of visible agents for each agent
        self.rule.update_visible_agents(self)

        # Generate the next message
        messages = await asyncio.gather(
            *[self.agents[i].astep(env_descriptions[i]) for i in agent_ids]
        )

        # Some rules will select certain messages from all the messages
        selected_messages = self.rule.select_message(self, messages)
        self.last_messages = selected_messages
        self.print_messages(selected_messages)

        # Update the memory of the agents
        self.rule.update_memory(self)

        # Update vision about other agents' position
        self.rule.updater.update_other_coordinate(self)

        self.cnt_turn += 1

        return selected_messages

    def print_messages(self, messages: List[Message]) -> None:
        for message in messages:
            if message is not None:
                logging.info(f"{message.sender}: {message.content}")

    def reset(self) -> None:
        # reset the cnt_turn
        self.cnt_turn = 0

        # reset the gird
        self.grids = np.zeros([self.grids_dim, self.grids_dim])
        for x_edge in range(self.grids_dim):
            self.grids[x_edge][0] = -1
            self.grids[x_edge][16] = -1
        for y_edge in range(self.grids_dim):
            self.grids[0][y_edge] = -1
            self.grids[16][y_edge] = -1

        # reset the rule
        self.rule.reset()

        # reset the enemy
        self.enemys = ["enemy_1", "enemy_2", "enemy_3", "enemy_4", "enemy_5"]

        # reset coordinates of enemys
        self.enemys_coordinate["enemy_1"] = [2,10]
        self.enemys_coordinate["enemy_2"] = [5,13]
        self.enemys_coordinate["enemy_3"] = [8,10]
        self.enemys_coordinate["enemy_4"] = [11,13]
        self.enemys_coordinate["enemy_5"] = [14,10]

        # reset health point of enemys
        self.enemys_healthpoint["enemy_1"] = 3
        self.enemys_healthpoint["enemy_2"] = 3
        self.enemys_healthpoint["enemy_3"] = 3
        self.enemys_healthpoint["enemy_4"] = 3
        self.enemys_healthpoint["enemy_5"] = 3

        # resurrect all the agents
        self.agents_survive["bot_1"] = 1
        self.agents_survive["bot_2"] = 1
        self.agents_survive["bot_3"] = 1
        self.agents_survive["bot_4"] = 1
        self.agents_survive["bot_5"] = 1

        # resurrect all the enemys
        self.enemys_survive["enemy_1"] = 1
        self.enemys_survive["enemy_2"] = 1
        self.enemys_survive["enemy_3"] = 1
        self.enemys_survive["enemy_4"] = 1
        self.enemys_survive["enemy_5"] = 1

        # reset grids occupancy
        self.grids_occupancy["2_10"] = "enemy_1"
        self.grids_occupancy["5_13"] = "enemy_2"
        self.grids_occupancy["8_10"] = "enemy_3"
        self.grids_occupancy["11_13"] = "enemy_4"
        self.grids_occupancy["14_10"] = "enemy_5"
        self.grids_occupancy["2_3"] = "bot_1"
        self.grids_occupancy["5_6"] = "bot_2"
        self.grids_occupancy["8_3"] = "bot_3"
        self.grids_occupancy["11_6"] = "bot_4"
        self.grids_occupancy["14_3"] = "bot_5"

        # reset the agents
        for agent in self.agents:
            agent.reset(environment=self)

    # check the end of the game
    def is_done(self) -> bool:
        """Check if the environment is done"""
        if (self.enemys_num == 0) or (self.agents_num == 0):
            return True
        else:
            return self.cnt_turn >= self.max_turns
