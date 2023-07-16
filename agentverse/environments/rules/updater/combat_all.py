from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

from . import updater_registry as UpdaterRegistry
from .basic import BasicUpdater
from agentverse.message import Message

if TYPE_CHECKING:
    from agentverse.environments import BaseEnvironment


@UpdaterRegistry.register("combat_all")
class CombatAllUpdater(BasicUpdater):
    def update_other_coordinate(self, environment: BaseEnvironment):
        # update agents' coordinates
        for agent in environment.agents:
            for other_agent in environment.agents:
                if agent.name == other_agent.name:
                    continue
                if other_agent.name in agent.others_coordinates:
                    if agent.others_coordinates[other_agent.name] != "Dead":
                        agent.others_coordinates[other_agent.name] = other_agent.coordinate
                else:
                    agent.others_coordinates[other_agent.name] = other_agent.coordinate
