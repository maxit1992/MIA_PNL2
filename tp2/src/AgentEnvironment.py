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
    context: list[dict[str, str]]
    answer: str
    chat_history: list[dict[str, str]]


class AgentEnvironment:

    def __init__(self, cv_agent1_file, cv_agent2_file, cv_agent3_file):
        # Input data
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
        graph.add_node("coordinator", self.init_and_get_required_agents)
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

    def init_and_get_required_agents(self, state: AgentState):
        greetings = {"role": "coordinator", "content": self.coordinator.greetings()}
        print(greetings)
        chat_history = [greetings]
        agents, prompt = self.coordinator.answer(state['question'],
                                                 [cv_agent['name'] for cv_agent in self.cv_agents_details])
        agent_answer = {"role": "coordinator",
                        "content": f"We need to ask the agents: {agents} about the question: {prompt}"}
        print(agent_answer)
        chat_history.append(agent_answer)
        return {'agents': agents, 'agents_prompt': prompt, 'context': [], 'chat_history': chat_history}

    def get_next_agent(self, state: AgentState):
        greetings = {"role": "selector", "content": "Hello I am the selector agent."
                                                    " I decide which CV agent should be asked next."}
        print(greetings)
        chat_history = state['chat_history']
        chat_history.append(greetings)
        agents = state['agents']
        if len(agents) > 0:
            next_agent = agents.pop(0)
        else:
            next_agent = "llm"
        agent_answer = {"role": "selector", "content": f"Next agent is {next_agent}"}
        print(agent_answer)
        chat_history.append(agent_answer)
        return {"agents": agents, "next_agent": next_agent, 'chat_history': chat_history}

    def select_agent(self, state: AgentState):
        return state['next_agent']

    def get_context_cv_agent1(self, state: AgentState):
        greetings = {"role": "agent1", "content": self.cv_agent1.greetings()}
        print(greetings)
        chat_history = state['chat_history']
        chat_history.append(greetings)
        answer = self.cv_agent1.answer(state['agents_prompt'])
        agent_answer = {"role": "agent1", "content": answer}
        print(agent_answer)
        chat_history.append(agent_answer)
        context = state['context']
        context.append({'candidate': self.cv_agents_details[0]['name'], 'context': answer})
        return {"context": context, "chat_history": chat_history}

    def get_context_cv_agent2(self, state: AgentState):
        greetings = {"role": "agent2", "content": self.cv_agent2.greetings()}
        print(greetings)
        chat_history = state['chat_history']
        chat_history.append(greetings)
        answer = self.cv_agent2.answer(state['agents_prompt'])
        agent_answer = {"role": "agent2", "content": answer}
        print(agent_answer)
        chat_history.append(agent_answer)
        context = state['context']
        context.append({'candidate': self.cv_agents_details[1]['name'], 'context': answer})
        return {"context": context, "chat_history": chat_history}

    def get_context_cv_agent3(self, state: AgentState):
        greetings = {"role": "agent3", "content": self.cv_agent3.greetings()}
        print(greetings)
        chat_history = state['chat_history']
        chat_history.append(greetings)
        answer = self.cv_agent3.answer(state['agents_prompt'])
        agent_answer = {"role": "agent3", "content": answer}
        print(agent_answer)
        chat_history.append(agent_answer)
        context = state['context']
        context.append({'candidate': self.cv_agents_details[2]['name'], 'context': answer})
        return {"context": context, "chat_history": chat_history}

    def answer_question(self, state: AgentState):
        greetings = {"role": "llm", "content": self.llm.greetings()}
        print(greetings)
        chat_history = state['chat_history']
        chat_history.append(greetings)
        context = state['context']
        answer = self.llm.answer(state['question'], context)
        agent_answer = {"role": "llm", "content": answer}
        print(agent_answer)
        chat_history.append(agent_answer)
        return {"answer": answer, "chat_history": chat_history}
