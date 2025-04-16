from typing import TypedDict

from langgraph.graph import StateGraph, END

from AgentCV import AgentCV
from AgentLLM import AgentLLM


class AgentState(TypedDict):
    question: str
    agents: list[str]
    next_agent: str
    agent_query: str
    messages: list[str]
    answer: str


class AgentEnvironment:

    def __init__(self):
        # Initialize the agents
        self.agent1 = AgentCV("Maxi Torti", "resources/CV.pdf")
        self.agent2 = AgentCV("Yann LeCun", "resources/CV2.pdf")
        self.agent3 = AgentCV("Andrew Ng", "resources/CV3.pdf")
        self.llm = AgentLLM()

        # Initialize the state graph
        graph = StateGraph(AgentState)
        graph.add_node("moderator", self.get_required_agents)
        graph.add_node("selector", self.get_next_agent)
        graph.add_node("agent1", self.get_context_agent1)
        graph.add_node("agent2", self.get_context_agent2)
        graph.add_node("agent3", self.get_context_agent3)
        graph.add_node("llm", self.answer_question)
        graph.add_edge("moderator", "selector")
        graph.add_conditional_edges("selector", self.select_agent)
        graph.add_edge("agent1", "selector")
        graph.add_edge("agent2", "selector")
        graph.add_edge("agent3", "selector")
        graph.add_edge("llm", END)
        graph.set_entry_point("moderator")
        self.graph = graph.compile()

    def get_required_agents(self, state: AgentState):
        print(f" get agents - {state}")
        user_question = state['question']
        agents = []
        if "Maxi Torti" in user_question:
            agents.append("agent1")
        if "Yann LeCun" in user_question:
            agents.append("agent2")
        if "Andrew Ng" in user_question:
            agents.append("agent3")
        print(f" get agents - {agents}")
        return {"agents": agents}

    def get_next_agent(self, state: AgentState):
        print(f"next_agent - {state}")
        agents = state['agents']
        if len(agents) > 0:
            next_agent = agents.pop(0)
        else:
            next_agent = "llm"
        print(f"next_agent - {next_agent}")
        return {"agents": agents, "next_agent": next_agent}

    def select_agent(self, state: AgentState):
        print(f"decide_agent - {state}")
        return state['next_agent']

    def get_context_agent1(self, state: AgentState):
        print(f"agent_1 - {state}")
        answer = self.agent1.answer(state['question'])
        print(f"agent_1 - {answer}")
        messages = state['messages'] if 'messages' in state else []
        messages.append(answer)
        return {"messages": messages}

    def get_context_agent2(self, state: AgentState):
        print(f"agent_2 - {state}")
        answer = self.agent2.answer(state['question'])
        print(f"agent_2 - {answer}")
        messages = state['messages'] if 'messages' in state else []
        messages.append(answer)
        return {"messages": messages}

    def get_context_agent3(self, state: AgentState):
        print(f"agent_3 - {state}")
        answer = self.agent3.answer(state['question'])
        print(f"agent_3 - {answer}")
        messages = state['messages'] if 'messages' in state else []
        messages.append(answer)
        return {"messages": messages}

    def answer_question(self, state: AgentState):
        context = state['messages'] if 'messages' in state else []
        answer = self.llm.answer(state['question'], context)
        return {"answer": answer}


abot = AgentEnvironment()
question = "Who has more experience in AI, Maxi Torti or Yann LeCun?"
result = abot.graph.invoke({"question": question})
print(result)
