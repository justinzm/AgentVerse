from __future__ import annotations

import re
from typing import Union

from agentverse.parser import OutputParser, LLMResult

# from langchain.schema import AgentAction, AgentFinish
from agentverse.utils import AgentAction, AgentFinish

from agentverse.parser import OutputParserError, output_parser_registry


@output_parser_registry.register("combat_task")
class CombatParser(OutputParser):
    def parse(self, output: LLMResult) -> Union[AgentAction, AgentFinish]:
        text = output.content
        cleaned_output = text.strip()
        cleaned_output = re.sub(r"\n+", "\n", cleaned_output)
        cleaned_output = cleaned_output.split("\n")
        if not (
            len(cleaned_output) == 2
            and cleaned_output[0].startswith("Thought:")
            and cleaned_output[1].startswith("Action:")
        ):
            raise OutputParserError(cleaned_output)
        thought = cleaned_output[0][len("Thought:") :].strip()
        action = cleaned_output[1][len("Action:") :].strip()
        return AgentFinish({"output": action}, text)
