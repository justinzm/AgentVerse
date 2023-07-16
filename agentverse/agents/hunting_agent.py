from __future__ import annotations

import logging
import bdb
from string import Template
from typing import TYPE_CHECKING, List
import re

from agentverse.message import Message

from . import agent_registry
from .base import BaseAgent

if TYPE_CHECKING:
    from agentverse.environments.base import BaseEnvironment

@agent_registry.register("hunting")
class HuntingAgent(BaseAgent):

    environment: BaseEnvironment = None
    coordinate: list = None
    prey_coordinates: dict = {}
    predator_coordinates: dict = {}


    def _fill_prompt_template(self, env_description: str = "") -> str:
        input_arguments = {
            "agent_name": self.name,
            "env_description": env_description,
            "role_description": self.role_description,
            "chat_history": self.memory.to_string(add_sender_prefix=True),
            "coordinate": self.coordinate,
            "prey_coordinates": self.prey_coordinates,
            "predator_coordinates": self.predator_coordinates
        }
        return Template(self.prompt_template).safe_substitute(input_arguments)


    def step(self, env_description: str = "") -> Message:
        prompt = self._fill_prompt_template(env_description)

        parsed_response = None
        for i in range(self.max_retry):
            try:
                response = self.llm.generate_response(prompt)
                parsed_response = self.output_parser.parse(response)
                break
            except KeyboardInterrupt:
                raise
            except Exception as e:
                logging.error(e)
                logging.warning("Retrying...")
                continue

        if parsed_response is None:
            logging.error(f"{self.name} failed to generate valid response.")

        message = Message(
            content=""
            if parsed_response is None
            else parsed_response.return_values["output"],
            sender=self.name,
            receiver=self.get_receiver(),
        )
        return message


    async def astep(self, env_description: str = "") -> Message:
        """Asynchronous version of step"""
        prompt = self._fill_prompt_template(env_description)

        parsed_response = None
        for i in range(self.max_retry):
            try:
                valid = True
                response = await self.llm.agenerate_response(prompt)
                parsed_response = self.output_parser.parse(response)
                # TODO : judgement (valid)
                break
            except (KeyboardInterrupt, bdb.BdbQuit):
                raise
            except Exception as e:
                logging.error(e)
                logging.warning("Retrying...")
                continue

        if not valid:
            # I don't know what to do, so I stay here and wait.
            parsed_response = None

        if parsed_response is None:
            logging.error(f"{self.name} failed to generate valid response.")

        message = Message(
            content="I don't know what to do, so I stay here and wait."
            if parsed_response is None
            else parsed_response.log,
            sender=self.name,
            receiver=self.get_receiver(),
        )
        return message


    def add_message_to_memory(self, messages: List[Message]) -> None:
        self.memory.add_message(messages)


    def reset(self, environment) -> None:
        """Reset the agent"""
        self.memory.reset()
        self.environment = environment

        # init agents' coordinates
        prey_list = []
        predator_list = []
        for agent in self.environment.agents:
            if agent.name != self.name:
                if self.environment.agents_role[agent.name] == 0:
                    prey_list.append(agent.name)
                else:
                    predator_list.append(agent.name)
        for prey in prey_list:
            self.prey_coordinates[prey] = "Not Known"
        for predator in predator_list:
            self.predator_coordinates[predator] = "Not Known"


    def _move(self, direction:str) -> bool:
        # update the coordinate
        if direction == "up":
            delta_x = -1
            delta_y = 0
        if direction == "down":
            delta_x = 1
            delta_y = 0
        if direction == "left":
            delta_x = 0
            delta_y = -1
        if direction == "right":
            delta_x = 0
            delta_y = 1
        cur_x, cur_y = self.coordinate

        # check move
        correctMove = True
        # There are already other bots in the location
        if f"{cur_x+delta_x}_{cur_y+delta_y}" in self.environment.grids_occupancy:
            if self.environment.grids_occupancy[f"{cur_x+delta_x}_{cur_y+delta_y}"] != "nobody":
                correctMove = False
                logging.error(f"{self.name}, you make a wrong move! There are already other bots at [{cur_x+delta_x}, {cur_y+delta_y}], so you have to stay().")
        # out of range
        if correctMove:
            if self.environment.grids[cur_x+delta_x][cur_y+delta_y] == -1:
                correctMove = False
                logging.error(f"{self.name}, you make a wrong move! [{cur_x+delta_x}, {cur_y+delta_y}] is out of range, so you have to stay().")
        
        # correct move
        if correctMove:

            # update grids_occupancy
            self.environment.grids_occupancy[f"{cur_x}_{cur_y}"] = "nobody"
            self.environment.grids_occupancy[f"{cur_x+delta_x}_{cur_y+delta_y}"] = self.name

            # update self coordinate
            self.coordinate[0] = cur_x + delta_x
            self.coordinate[1] = cur_y + delta_y

            # update coordinate in others' sights
            for other_agent in self.environment.agents:
                if other_agent.name != self.name:
                    if self.environment.agents_role[self.name] == 0:
                        other_agent.prey_coordinates[self.name] = self.coordinate
                    else:
                        other_agent.predator_coordinates[self.name] = self.coordinate

        return correctMove


    def _hunt(self, target:str) -> bool:
        # prey cannot hunt
        if self.environment.agents_role[self.name] == 0:
            logging.error(f"{self.name}, you cannot hunt others as a prey, so you have to stay().")
            return False
        if self.environment.agents_role[target] == 1:
            logging.error(f"{self.name}, you make a wrong hunting! {target} is a predator, so you have to stay().")
            return False
        # predator hunt the target
        else:
            # check hunt
            inRange = False
            for x in [self.coordinate[0] - 1, self.coordinate[0], self.coordinate[0] + 1]:
                if inRange:
                    break
                for y in [self.coordinate[1] - 1, self.coordinate[1], self.coordinate[1] + 1]:
                    if f"{x}_{y}" in self.environment.grids_occupancy:
                        if self.environment.grids_occupancy[f"{x}_{y}"] == target:
                            inRange = True
                            self.environment.agents_survive[target] = 0
                            self.environment.prey_num -= 1
                            self.environment.grids_occupancy[f"{x}_{y}"] = "nobody"
                            for agent in self.environment.agents:
                                agent.prey_coordinates[target] = "Dead"
                            break
            if not inRange:
                logging.error(f"{self.name}, you make a wrong hunting! {target} is out of your hunting range, so you have to stay().")
            return inRange


    def _stay(self) -> bool:
        return True
