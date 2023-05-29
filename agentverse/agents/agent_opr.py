"""
An agent based upon Observation-Planning-Reflection architecture.
"""

from logging import getLogger

from abc import abstractmethod
from typing import List, Set, Union, NamedTuple, TYPE_CHECKING

from pydantic import BaseModel, Field, validator

from agentverse.llms import BaseLLM
from agentverse.memory import BaseMemory, ChatHistoryMemory
from agentverse.message import Message
from agentverse.parser import OutputParser

from agentverse.message import Message
from agentverse.agents.base import BaseAgent

# from agentverse.utils.prompts

from datetime import datetime as dt
import datetime

from . import agent_registry
from string import Template


logger = getLogger(__file__)

if TYPE_CHECKING:
    from agentverse.environments.base import BaseEnvironment


@agent_registry.register("OPR")
class AgentOPR(BaseAgent):
    async_mode: bool = True,
    current_time: str = None,
    traits: str = None,
    whole_day_plan: dict = Field(default_factory=dict)
    environment: "BaseEnvironment" = None
    step_cnt: int = 0
    summary_interval: int = 10
    reflection_interval: int = 10

    status: str = Field(default=None, description="what the agent is doing according to whole_day_plan")
    status_start_time: dt = Field(default=None)
    status_duration: int = Field(default=0,
                                      description="we use this field and current time to check when to get_plan in func:`check_status_passive` ")


    @validator('current_time')
    def convert_str_to_dt(cls, current_time):
        if not isinstance(current_time, str):
            raise ValueError('current_time should be str')
        return dt.strptime(current_time, "%Y-%m-%d %H:%M:%S")

    def step(self, current_time: dt, env_description: str = "") -> Message:
        """
        Call this method at each time frame
        """
        self.current_time = current_time

        logger.debug(
            "Agent {}, Time: {}, Status {}, Status Start: {}, Will last: {}".format(
                self.state_dict["name"],
                str(self.current_time),
                self.status,
                self.status_start_time,
                datetime.timedelta(seconds=self.status_duration),
            )
        )

        # To ensure the proper functioning of the agent, the memory, plan, and summary cannot be empty. Therefore, it is necessary to perform an initialization similar to what should be done at the beginning of each day.
        # self.minimal_init()

        # before we handle any observation, we first check the status.
        # self.check_status_passive()

        # self.observe()

        # if self.might_react():
        #     self.react()
        #
        # if self.movement:
        #     self.analysis_movement_target(self.movement_description)
        #
        # # 3.5 add observation to memory
        # for ob in self.incoming_observation:
        #     self.long_term_memory.add(ob, self.current_time, ["observation"])
        # self.incoming_observation = []  # empty the incoming observation

        # 4. Periodic fixed work of reflection and summary (tentatively set to be done every 100 logical frames).

    # TODO chimin

    def check_status_passive(self, ):
        """Check if the current status needs to be finished. If so, examine the plan and initiate the next action.
        """
        if self.status_start_time is None: # fixing empty start time
            self.status_start_time = self.current_time

        if self.status_start_time+datetime.timedelta(self.status_duration) <= self.current_time:
            next_plan = self.memory.planner.get_plan(current_time=self.current_time)
            self.status_start_time = self.current_time
            self.status = next_plan['status']
            self.status_duration = next_plan['duration']
        else:
            logger.debug(f"{self.name} don't change status by plan: {self.status_start_time}, {datetime.timedelta(self.status_duration)}, {self.current_time}")

    async def astep(self, current_time: dt,env_description: str = "") -> Message:
        """Asynchronous version of step"""
        #use environment's time to update agent's time
        self.current_time = current_time
        # Before the agent step, we check current status,
        self.check_status_passive()

        prompt = self._fill_prompt_template(env_description)

        parsed_response = None
        for i in range(self.max_retry):
            try:
                response = await self.llm.agenerate_response(prompt)
                parsed_response = self.output_parser.parse(response)

                if 'say(' in parsed_response.return_values["output"]:
                    reaction, target = eval("self._" + parsed_response.return_values["output"].strip())
                elif 'act(' in parsed_response.return_values["output"]:
                    reaction, target = eval("self._" + parsed_response.return_values["output"].strip())
                elif 'do_nothing(' in parsed_response.return_values["output"]:
                    reaction, target = None, None

                break

            except Exception as e:
                logger.error(e)
                logger.warning("Retrying...")
                continue

        if parsed_response is None:
            logger.error(f"{self.name} failed to generate valid response.")




        message = Message(
            content=""
            if reaction is None
            else reaction,
            sender=self.name,
            receiver=self.get_receiver() if target is None else target,
        )

        # TODO currently, summary is not added back to memory while reflection is
        self.step_cnt += 1

        if self.step_cnt % self.summary_interval == 0:
            self.memory.summary = self.memory.generate_summary(self.current_time)

        if self.step_cnt % self.reflection_interval == 0:
            _ = self.reflect(self.current_time)

        return message

    def _act(self, description=None, target=None):
        if description is None:
            return ""
        if target is None:
            reaction_content = f"{self.name} performs action: '{description}'."
        else:
            reaction_content = f"{self.name} performs action to {target}: '{description}'."
        # self.environment.broadcast_observations(self, target, reaction_content)


        return reaction_content, target

    def _say(self, description, target=None):
        if description is None:
            return ""
        if target is None:
            reaction_content = f"{self.name} says: '{description}'."
        else:
            reaction_content = f"{self.name} says to {target}: '{description}'."
        # self.environment.broadcast_observations(self, target, reaction_content)
        return reaction_content, target


    def _fill_prompt_template(self, env_description: str = "") -> str:
        """Fill the placeholders in the prompt template

        In the conversation agent, three placeholders are supported:
        - ${agent_name}: the name of the agent
        - ${env_description}: the description of the environment
        - ${role_description}: the description of the role of the agent
        - ${chat_history}: the chat history of the agent
        """
        input_arguments = {
            "agent_name": self.name,
            "summary": self.memory.summary,
            "current_time": self.current_time,
            "status": self.status,
            "env_description": env_description,
        }
        return Template(self.prompt_template).safe_substitute(input_arguments)

    # TODO call longtermmemory element
    def add_message_to_memory(self, messages: List[Message]) -> None:
        for message in messages:
            self.memory.add_message(message, time=self.current_time)

    # Should call this when status changed, plan==status
    def add_plan_to_memory(self,) -> None:
        self.memory.add_plan(content=self.status, time=self.current_time)

    def reset(self, environment: "BaseEnvironment") -> None:
        """Reset the agent"""
        self.environment = environment

        self.memory.reset(environment=environment, agent=self)
        # TODO: reset receiver
