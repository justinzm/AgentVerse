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

@agent_registry.register("combat")
class CombatAgent(BaseAgent):

    environment: BaseEnvironment = None
    coordinate: list = None
    others_coordinates: dict = {}
    enemys_coordinates: dict = {}
    health_point: int = 3
    attack_ability: int = 1


    def _fill_prompt_template(self, env_description: str = "") -> str:
        input_arguments = {
            "agent_name": self.name,
            "env_description": env_description,
            "role_description": self.role_description,
            "chat_history": self.memory.to_string(add_sender_prefix=True),
            "coordinate": self.coordinate,
            "others_coordinates": self.others_coordinates,
            "enemys_coordinates": self.enemys_coordinates
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
                if "move" in parsed_response.return_values["output"]:
                    direction = re.search(re.compile('move\((.*?)\)'), parsed_response.return_values["output"]).group(1)
                    valid = self._move(direction)
                elif "attack" in parsed_response.return_values["output"]:
                    target = re.search(re.compile('attack\((.*?)\)'), parsed_response.return_values["output"]).group(1)
                    valid = self._attack(target)
                elif "stay" in parsed_response.return_values["output"]:
                    valid = self._stay()
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

        # init others angents' coordinates
        others_agents = []
        for agent in self.environment.agents:
            if agent.name != self.name:
                others_agents.append(agent.name)
        for others_agent in others_agents:
            self.others_coordinates[others_agent] = "Not Known"
        
        # init enemy bots' coordinates
        for enemy in self.environment.enemys:
            self.enemys_coordinates[enemy] = "Not Known"

        # initiate health point
        self.health_point = 3


    def _move(self, direction:str) -> bool:
        # restore attack_ability
        self.attack_ability = 1

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

            # update coordinates in others' sights
            for other_agent in self.environment.agents:
                if other_agent.name != self.name:
                    other_agent.others_coordinates[self.name] = self.coordinate

        return correctMove


    def _attack(self, target:str) -> bool:
        # attack the target
        inRange = False
        for x in [self.coordinate[0] - 1, self.coordinate[0], self.coordinate[0] + 1]:
            if inRange:
                break
            for y in [self.coordinate[1] - 1, self.coordinate[1], self.coordinate[1] + 1]:
                if f"{x}_{y}" in self.environment.grids_occupancy:
                    if self.environment.grids_occupancy[f"{x}_{y}"] == target:
                        inRange = True
                        self.environment.enemys_healthpoint[target] -= 1
                        # about die
                        if self.environment.enemys_healthpoint[target] == 0:
                            logging.info(f"{target} is dead!")
                            self.environment.enemys_num -= 1
                            self.environment.enemys_coordinate[target] = "Dead"
                            self.enemys_coordinates[target] = "Dead"
                            self.environment.enemys_survive[target] = 0
                            self.environment.grids_occupancy[f"{x}_{y}"] = "nobody"
                        break
        if not inRange:
            logging.error(f"{self.name}, you make a wrong attack! {target} is out of your firing range, so you have to stay().")

        # lose attack_ability
        self.attack_ability = 0

        return inRange


    def _stay(self) -> bool:
        # restore attack_ability
        self.attack_ability = 1

        return True
