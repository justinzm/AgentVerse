import os

os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"
os.environ["all_proxy"] = "socks5://127.0.0.1:7890"


import datetime
import logging
import random
import numpy as np

from typing import Any, List, Optional, Union
from datetime import datetime as dt
from sklearn.metrics.pairwise import cosine_similarity
from pydantic import Field

from agentverse.llms.openai import get_embedding, chat
from agentverse.memory.base import BaseMemory
from agentverse.agents import BaseAgent
from agentverse.environments.base import BaseEnvironment
from agentverse.memory.memory_element.LongtermMemoryElement import LongtermMemoryElement
from agentverse.memory.memory_element.Reflection import Reflection
from agentverse.message import Message
from agentverse.memory.memory_element.BaseMemoryElement import BaseMemoryElement
from agentverse.memory.utils.Planner import Planner

IMPORTANCE_PROMPT = """On the scale of 1 to 10, where 1 is purely mundane \
(e.g., brushing teeth, making bed) and 10 is \
extremely poignant (e.g., a break up, college \
acceptance), rate the likely poignancy of the \
following piece of memory. \
If you think it's too hard to rate it, you can give an inaccurate assessment. \
The content or people mentioned is not real. You can hypothesis any reasonable context. \
Please strictly only output one number. \
Memory: {} \
Rating: <fill in>"""
IMMEDIACY_PROMPT = """On the scale of 1 to 10, where 1 is requiring no short time attention\
(e.g., a bed is in the room) and 10 is \
needing quick attention or immediate response(e.g., being required a reply by others), rate the likely immediacy of the \
following statement. \
If you think it's too hard to rate it, you can give an inaccurate assessment. \
The content or people mentioned is not real. You can hypothesis any reasonable context. \
Please strictly only output one number. \
Memory: {} \
Rating: <fill in>"""
QUESTION_PROMPT = """Given only the information above, what are 3 most salient \
high-level questions we can answer about the subjects in the statements?"""

INSIGHT_PROMPT = """What at most 5 high-level insights can you infer from \
the above statements? Only output insights with high confidence. 
example format: insight (because of 1, 5, 3)"""


def get_questions(texts):
    prompt = "\n".join(texts) + "\n" + QUESTION_PROMPT
    result = chat(prompt)
    questions = [q for q in result.split("\n") if len(q.strip()) > 0]
    questions = questions[:3]
    return questions


def get_insights(statements):
    prompt = ""
    for i, st in enumerate(statements):
        prompt += str(i + 1) + ". " + st + "\n"
    prompt += INSIGHT_PROMPT
    result = chat(prompt)
    insights = [isg for isg in result.split("\n") if len(isg.strip()) > 0][:5]
    insights = [".".join(i.split(".")[1:]) for i in insights]
    # remove insight pointers for now
    insights = [i.split("(")[0].strip() for i in insights]
    return insights


class ReflectionMemory(BaseMemory):
    # on load, load our database
    """
    memory_index: path for saving memory json file
    importance_threshold: the threshold for deciding whether to do reflection

    """
    agent: Optional[BaseAgent] = None
    environment: Optional[BaseEnvironment] = None
    importance_threshold: int = Field(default=100)
    memories: List[LongtermMemoryElement] = Field(default_factory=list)
    planner: Planner = None
    accumulated_importance: int = Field(default=0)

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        clear_memory = True  # TODO: add this to arguments

        # the least importance threshold for reflection.
        # TODO: add none-default value in the yaml config file
        self.importance_threshold = getattr(
            self.agent, "importance_threshold", self.importance_threshold
        )

        self.memories = []

        # TODO add argument in agent - (daily_plans, current_time)
        self.planner = Planner(daily_plans=list(self.agent.whole_day_plan.values())[0],
                               agent=self.agent,
                               current_time=self.agent.current_time,
                               environment=self.environment)
        self.add_plan(content=self.planner.get_whole_day_plan_text(), time=self.agent.current_time)

        # TODO: load last time memory from file
        # currently, we just initialize blank memory

        self.accumulated_importance = 0
        if len(self.memories) > 0:
            for m in self.memories:
                if isinstance(m, Reflection):
                    break
                self.accumulated_importance += m.importance


    def add_message(self, message: Message, time: dt) -> None:
        """
        Add a message into longterm memory as LongtermMemory object.
        Overload with the BaseMemory.add_message with more arguments(time: dt)
        """
        self.add_memory(
            LongtermMemoryElement.create_from_message(
                message=message,
                subject=self.agent,
                time=time,
            )
        )

    def add_plan(self, content: str, time: dt) -> None:

        self.add_memory(
            LongtermMemoryElement.create_longterm_memory(
                content=content,
                subject=self.agent,
                time=time,
            )
        )

    def reset(self) -> None:
        self.memories = []

    def add_memory(self, memory: LongtermMemoryElement) -> None:
        """
        Add memory element to ReflectionMemory
        """
        # TODO: check duplicate memories(same content, same time)
        # TODO: dump to json file

        self.memories.append(memory)
        if isinstance(memory, Reflection):
            self.accumulated_importance = 0
        else:
            self.accumulated_importance += memory.importance

    def get_memory(
        self, content: str, current_time, cnt_retrieved_entries: int = 1
    ) -> Optional[List[Any]]:
        """
        Get k-most relevant memories to content
        """
        return self.query(content, cnt_retrieved_entries, current_time)

    def query(
        self, text: Union[str, List[str]], k: int, current_time: dt, nms_threshold=0.99
    ) -> List[str]:
        """
        get top-k entry based on recency, relevance, importance, immediacy
        The query result can be Short-term or Long-term queried result.
        formula is
        $$ score= sim(q,v) *max(LTM\_score, STM\_score) $$
        $$ STM\_score=time\_score(createTime)*immediacy $$
        $$ LTM\_score=time\_score(accessTime)*importance $$
        time score is exponential decay weight. stm decays faster.

        The query supports querying based on multiple texts and only gives non-overlapping results
        If nms_threshold is not 1, nms mechanism if activated. By default,
        use soft nms with modified iou base(score starts to decay iff cos sim is higher than this value,
         and decay weight at this value if 0. rather than 1-threshold).


        Args:
            text: str
            k: int

        Returns: List[str]
        """
        assert len(text) > 0
        texts = [text] if isinstance(text, str) else text
        maximum_score = None
        for text in texts:
            embedding = get_embedding(text)
            score = []
            for memory in self.memories:
                last_access_time_diff = (
                    current_time - memory.last_access_time
                ).total_seconds() // 3600
                recency = np.power(
                    0.99, last_access_time_diff
                )  # TODO: review the metaparameter 0.99

                create_time_diff = (
                    current_time - memory.create_time
                ).total_seconds() // 60
                instancy = np.power(
                    0.90, create_time_diff
                )  # TODO: review the metaparameter 0.90

                relevance = cosine_similarity(
                    np.array(embedding).reshape(1, -1),
                    np.array(memory.embedding).reshape(1, -1),
                )[0][0]

                importance = memory.importance / 10
                immediacy = memory.immediacy / 10

                ltm_w = recency * importance
                stm_w = instancy * immediacy

                score.append(relevance * np.maximum(ltm_w, stm_w))

            score = np.array(score)

            if maximum_score is not None:
                maximum_score = np.maximum(score, maximum_score)
            else:
                maximum_score = score

        if nms_threshold == 1:
            # no nms is triggered
            top_k_indices = np.argsort(maximum_score)[-k:][::-1]
        else:
            # TODO: soft-nms
            assert 0 <= nms_threshold < 1
            top_k_indices = []
            while len(top_k_indices) < min(k, len(self.memories)):
                top_index = np.argmax(maximum_score)
                top_k_indices.append(top_index)
                maximum_score[top_index] = -1  # anything to prevent being chosen again
                top_embedding = self.memories[top_index].embedding
                cos_sim = cosine_similarity(
                    np.array(top_embedding).reshape(1, -1),
                    np.array([memory.embedding for memory in self.memories]),
                )[0]
                score_weight = np.ones_like(maximum_score)
                score_weight[cos_sim >= nms_threshold] -= (
                    cos_sim[cos_sim >= nms_threshold] - nms_threshold
                ) / (1 - nms_threshold)
                maximum_score = maximum_score * score_weight

        # access them and refresh the access time
        for i in top_k_indices:
            self.memories[i].last_access_time = current_time
        # sort them in time periods. if the data tag is 'observation', ad time info output.
        top_k_indices = sorted(
            top_k_indices, key=lambda k: self.memories[k].create_time
        )
        query_results = []
        for i in top_k_indices:
            query_result = self.memories[i].content
            query_results.append(query_result)
        return query_results

    def reflect(self, time: dt):
        """
        initiate a reflection that inserts high level knowledge to memory
        """
        # check if importance exceeds the threshold
        should_reflect = self.accumulated_importance >= self.importance_threshold
        if not should_reflect:
            logging.debug(
                f"Doesn't reflect since accumulated_importance={self.accumulated_importance} < reflection_threshold={self.importance_threshold}"
            )
            return "reflection reject: prevent duplicate reflecting result"
        if not self.memories:
            return "reflection reject: empty memories"

        memories_of_interest = self.memories[-100:]
        questions = get_questions([m.content for m in memories_of_interest])
        statements = self.query(questions, len(questions) * 10, time)
        insights = get_insights(statements)
        logging.info(self.agent.name + f" Insights: {insights}")
        for insight in insights:
            self.add_memory(
                Reflection.create_longterm_memory(
                    content=insight,
                    time=time,
                    subject=self.agent,
                )
            )  # This will add a Reflection instance instead of LongtermMemory instance
        return insights

    def __repr__(self) -> str:
        memory_string = "\n".join([str(memory) for memory in self.memories])
        return f"ReflectionMemory({memory_string})"

    def to_string(self) -> str:
        return self.__repr__()


if __name__ == "__main__":
    from agentverse.initialization import load_agent, load_environment, prepare_task_config

    task_config = prepare_task_config("alice_home")
    agents = []
    for agent_configs in task_config["agents"]:
        agent = load_agent(agent_configs)
        agents.append(agent)

    # Build the environment
    env_config = task_config["environment"]
    env_config["agents"] = agents
    environment = load_environment(env_config)
    memory = ReflectionMemory(agent=agents[0], environment=environment)
    # get next plan
    next_plan = memory.planner.get_next_plan()
    next_next_plan = memory.planner.get_next_plan()

    message_list = [
        Message(content="I am a student"),
        Message(content="I am drunk"),
        Message(content="My girlfriend is staring at me"),
        Message(
            content="Teacher looks at me with an approving smile.",
        ),
        Message(content="A dagger is stuck into my heart"),
        Message(content="The TV is on."),
        Message(content="Jane sniffs at me"),
        Message(
            content="Bob carefully listed out all the TODOs.",
        ),
        Message(content="Prof. Liu applauded with my work"),
        Message(content="I am falling into the hell"),
    ]

    for m in message_list:
        memory.add_message(m, dt.now() - datetime.timedelta(hours=random.random() * 10))

    memory.reflect(dt.now())
    print(memory.query("What is my mood now", 2, dt.now()))

    with open("./logging/reflection_unit_test_log.log", "w") as fp:
        import json

        json.dump(memory, fp, indent=4, default=str)