from typing import TypedDict

from langgraph.graph import StateGraph, END


class AgentState(TypedDict):
    question: str
    agents: list[str]
    next_agent: str
    messages: list[str]
    answer: str


class System:

    def __init__(self):
        graph = StateGraph(AgentState)
        graph.add_node("moderator", self.get_agents)
        graph.add_node("selector", self.next_agent)
        graph.add_node("agent1", self.agent1)
        graph.add_node("agent2", self.agent2)
        graph.add_node("agent3", self.agent3)
        graph.add_node("llm", self.answer)
        graph.add_edge("moderator", "selector")
        graph.add_conditional_edges("selector", self.decide_agent,
                                    {"agent1": "agent1", "agent2": "agent2", "agent3": "agent3", "llm": "llm"})
        graph.add_edge("agent1", "selector")
        graph.add_edge("agent2", "selector")
        graph.add_edge("agent3", "selector")
        graph.add_edge("llm", END)
        graph.set_entry_point("moderator")
        self.graph = graph.compile()

    def get_agents(self, state: AgentState):
        print(f" get agents - {state}")
        question = state['question']
        agents = []
        if "agent1" in question:
            agents.append("agent1")
        if "agent2" in question:
            agents.append("agent2")
        if "agent3" in question:
            agents.append("agent3")
        print(f" get agents - {agents}")
        return {"agents": agents}

    def next_agent(self, state: AgentState):
        print(f"next_agent - {state}")
        agents = state['agents']
        if len(agents) > 0:
            next_agent = agents.pop(0)
        else:
            next_agent = "llm"
        print(f"next_agent - {next_agent}")
        return {"agents": agents, "next_agent": next_agent}

    def decide_agent(self, state: AgentState):
        print(f"decide_agent - {state}")
        return state['next_agent']

    def agent1(self, state: AgentState):
        print(f"agent_1 - {state}")
        messages = state['messages'] if 'messages' in state else []
        messages.append("{agent:1, content:\"26 years\"}")
        return {"messages": messages}

    def agent2(self, state: AgentState):
        print(f"agent_2 - {state}")
        messages = state['messages'] if 'messages' in state else []
        messages.append("{agent:2, content:\"45 years\"}")
        return {"messages": messages}

    def agent3(self, state: AgentState):
        print(f"agent_3 - {state}")
        messages = state['messages'] if 'messages' in state else []
        messages.append("{agent:3, content:\"10 years\"}")
        return {"messages": messages}

    def answer(self, state: AgentState):
        answer = 'NI IDEA'
        return {"answer": answer}


abot = System()
question = "cuantos años tiene el agent1 y el agent3?"
result = abot.graph.invoke({"question": question})
print(result)
