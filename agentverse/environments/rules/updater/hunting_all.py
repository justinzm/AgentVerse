from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

from . import updater_registry as UpdaterRegistry
from .basic import BasicUpdater
from agentverse.message import Message

if TYPE_CHECKING:
    from agentverse.environments import BaseEnvironment


@UpdaterRegistry.register("hunting_all")
class HuntingAllUpdater(BasicUpdater):
    def update_other_coordinate(self, environment: BaseEnvironment):
        # update agents' coordinates
        for agent in environment.agents:
            for other_agent in environment.agents:
                if agent.name == other_agent.name:
                    continue
                else:
                    if other_agent.name in agent.prey_coordinates:
                        if agent.prey_coordinates[other_agent.name] != "Dead":
                            if environment.agents_role[other_agent.name] == 0:
                                agent.prey_coordinates[other_agent.name] = other_agent.coordinate
                            else:
                                agent.predator_coordinates[other_agent.name] = other_agent.coordinate
                    else:
                        if environment.agents_role[other_agent.name] == 0:
                                agent.prey_coordinates[other_agent.name] = other_agent.coordinate
                        else:
                            agent.predator_coordinates[other_agent.name] = other_agent.coordinate

