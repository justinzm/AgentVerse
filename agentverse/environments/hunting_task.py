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


@EnvironmentRegistry.register("hunting")
class HuntingEnvironment(BaseEnvironment):
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
    # key: name of agent, value: agent survives or not
    agents_survive: defaultdict = defaultdict(int)
    # key: name of agent, value: agent is predator or not
    agents_role: defaultdict = defaultdict(int)
    # number of prey which survive
    prey_num: int = 3
    max_turns: int = 10
    cnt_turn: int = 0
    grids: np.ndarray = None
    # 12 = 1 + 10 + 1 (including padding * 2)
    grids_dim: int = 12
    # key: coordinate, value: name of bot
    grids_occupancy: defaultdict = defaultdict(str)
    last_messages: List[Message] = []
    rule_params: Dict = {}

    def __init__(self, rule, **kwargs):
        # all agents survive at first
        tmp_agents_survive = {}
        tmp_agents_survive["prey_1"] = 1
        tmp_agents_survive["prey_2"] = 1
        tmp_agents_survive["prey_3"] = 1
        tmp_agents_survive["predator_1"] = 1
        tmp_agents_survive["predator_2"] = 1
        tmp_agents_survive["predator_3"] = 1

        # init agents' role
        tmp_agents_role = {}
        tmp_agents_role["prey_1"] = 0
        tmp_agents_role["prey_2"] = 0
        tmp_agents_role["prey_3"] = 0
        tmp_agents_role["predator_1"] = 1
        tmp_agents_role["predator_2"] = 1
        tmp_agents_role["predator_3"] = 1

        # set grids occupancy
        tmp_grids_occupancy = {}
        tmp_grids_occupancy["3_2"] = "predator_1"
        tmp_grids_occupancy["6_4"] = "predator_2"
        tmp_grids_occupancy["8_2"] = "predator_3"
        tmp_grids_occupancy["3_7"] = "prey_1"
        tmp_grids_occupancy["6_9"] = "prey_2"
        tmp_grids_occupancy["8_7"] = "prey_3"

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
            agents_survive = tmp_agents_survive,
            agents_role = tmp_agents_role,
            grids_occupancy = tmp_grids_occupancy,
            **kwargs
            )

    async def step(self) -> List[Message]:
        """Run one step of the environment"""
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
            self.grids[x_edge][11] = -1
        for y_edge in range(self.grids_dim):
            self.grids[0][y_edge] = -1
            self.grids[11][y_edge] = -1

        # reset the rule
        self.rule.reset()

        # resurrect all the agents
        self.agents_survive["prey_1"] = 1
        self.agents_survive["prey_2"] = 1
        self.agents_survive["prey_3"] = 1
        self.agents_survive["predator_1"] = 1
        self.agents_survive["predator_2"] = 1
        self.agents_survive["predator_3"] = 1

        # reset agents' role
        self.agents_role["prey_1"] = 0
        self.agents_role["prey_2"] = 0
        self.agents_role["prey_3"] = 0
        self.agents_role["predator_1"] = 1
        self.agents_role["predator_2"] = 1
        self.agents_role["predator_3"] = 1

        # reset grids occupancy
        self.grids_occupancy["3_2"] = "predator_1"
        self.grids_occupancy["6_4"] = "predator_2"
        self.grids_occupancy["8_2"] = "predator_3"
        self.grids_occupancy["3_7"] = "prey_1"
        self.grids_occupancy["6_9"] = "prey_2"
        self.grids_occupancy["8_7"] = "prey_3"

        # reset the agents
        for agent in self.agents:
            agent.reset(environment=self)

    # check the end of the game
    def is_done(self) -> bool:
        """Check if the environment is done"""
        if self.prey_num == 0:
            return True
        else:
            return self.cnt_turn >= self.max_turns
