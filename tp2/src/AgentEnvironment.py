from pathlib import Path
from typing import TypedDict

from langgraph.graph import StateGraph, END

from AgentCV import AgentCV
from AgentCoordinator import AgentCoordinator
from AgentLLM import AgentLLM


class AgentState(TypedDict):
    question: str
    agents: list[str]
    agents_prompt: str
    next_agent: str
    messages: list[str]
    answer: str


class AgentEnvironment:

    def __init__(self):
        # Input data
        cv_agent1_file = "resources/Maxi Torti.pdf"
        cv_agent2_file = "resources/Yann LeCun.pdf"
        cv_agent3_file = "resources/Andrew Ng.pdf"
        self.cv_agents_details = []
        self.cv_agents_details.append({'name': Path(cv_agent1_file).stem, 'file': cv_agent1_file})
        self.cv_agents_details.append({'name': Path(cv_agent2_file).stem, 'file': cv_agent2_file})
        self.cv_agents_details.append({'name': Path(cv_agent3_file).stem, 'file': cv_agent3_file})

        # Initialize the agents
        self.coordinator = AgentCoordinator()
        self.cv_agent1 = AgentCV(self.cv_agents_details[0]['name'], self.cv_agents_details[0]['file'])
        self.cv_agent2 = AgentCV(self.cv_agents_details[1]['name'], self.cv_agents_details[1]['file'])
        self.cv_agent3 = AgentCV(self.cv_agents_details[2]['name'], self.cv_agents_details[2]['file'])
        self.llm = AgentLLM()

        # Initialize the state graph
        graph = StateGraph(AgentState)
        graph.add_node("coordinator", self.get_required_agents)
        graph.add_node("selector", self.get_next_agent)
        graph.add_node(self.cv_agents_details[0]['name'], self.get_context_cv_agent1)
        graph.add_node(self.cv_agents_details[1]['name'], self.get_context_cv_agent2)
        graph.add_node(self.cv_agents_details[2]['name'], self.get_context_cv_agent3)
        graph.add_node("llm", self.answer_question)
        graph.add_edge("coordinator", "selector")
        graph.add_conditional_edges("selector", self.select_agent)
        graph.add_edge(self.cv_agents_details[0]['name'], "selector")
        graph.add_edge(self.cv_agents_details[1]['name'], "selector")
        graph.add_edge(self.cv_agents_details[2]['name'], "selector")
        graph.add_edge("llm", END)
        graph.set_entry_point("coordinator")
        self.graph = graph.compile()

    def get_required_agents(self, state: AgentState):
        print(self.coordinator.greetings())
        user_question = state['question']
        coordinator_answer = self.coordinator.answer(user_question,
                                                     [cv_agent['name'] for cv_agent in self.cv_agents_details])
        print(f"coordinator - {coordinator_answer}")
        return coordinator_answer

    def get_next_agent(self, state: AgentState):
        print("Hello I am the selector agent. I decide which CV agent should be asked next.")
        agents = state['agents']
        if len(agents) > 0:
            next_agent = agents.pop(0)
        else:
            next_agent = "llm"
        print(f"next_agent - {next_agent}")
        return {"agents": agents, "next_agent": next_agent}

    def select_agent(self, state: AgentState):
        return state['next_agent']

    def get_context_cv_agent1(self, state: AgentState):
        print(self.cv_agent1.greetings())
        answer = self.cv_agent1.answer(state['agents_prompt'])
        print(f"cv_agent_1 - {answer}")
        messages = state['messages'] if 'messages' in state else []
        messages.append(answer)
        return {"messages": messages}

    def get_context_cv_agent2(self, state: AgentState):
        print(self.cv_agent2.greetings())
        answer = self.cv_agent2.answer(state['agents_prompt'])
        print(f"cv_agent_2 - {answer}")
        messages = state['messages'] if 'messages' in state else []
        messages.append(answer)
        return {"messages": messages}

    def get_context_cv_agent3(self, state: AgentState):
        print(self.cv_agent3.greetings())
        answer = self.cv_agent3.answer(state['agents_prompt'])
        print(f"cv_agent_3 - {answer}")
        messages = state['messages'] if 'messages' in state else []
        messages.append(answer)
        return {"messages": messages}

    def answer_question(self, state: AgentState):
        print(self.llm.greetings())
        context = state['messages'] if 'messages' in state else []
        answer = self.llm.answer(state['question'], context)
        print(f"llm - {answer}")
        return {"answer": answer}


abot = AgentEnvironment()
question = "Who has more experience in AI, Maxi Torti or Yann LeCun?"
question = "Who is older among all candidates?"
result = abot.graph.invoke({"question": question})
print(f"result = {result}")
