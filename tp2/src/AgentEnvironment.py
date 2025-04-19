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
    """
    This class manages the environment for coordinating multiple agents to answer user questions based on CV data.
    It initializes the agents, sets up a state graph, and orchestrates the flow of information between agents.
    """

    def __init__(self, cv_agent1_file, cv_agent2_file, cv_agent3_file):
        """
        Initializes the AgentEnvironment with the provided CV files, sets up agents, and compiles the state graph.

        Args:
            cv_agent1_file (str): The file path for the first CV.
            cv_agent2_file (str): The file path for the second CV.
            cv_agent3_file (str): The file path for the third CV.
        """
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
        graph.add_node("coordinator", self._init_and_get_required_agents)
        graph.add_node("selector", self._get_next_agent)
        graph.add_node(self.cv_agents_details[0]['name'], self._get_context_cv_agent1)
        graph.add_node(self.cv_agents_details[1]['name'], self._get_context_cv_agent2)
        graph.add_node(self.cv_agents_details[2]['name'], self._get_context_cv_agent3)
        graph.add_node("llm", self._answer_question)
        graph.add_edge("coordinator", "selector")
        graph.add_conditional_edges("selector", self._select_agent)
        graph.add_edge(self.cv_agents_details[0]['name'], "selector")
        graph.add_edge(self.cv_agents_details[1]['name'], "selector")
        graph.add_edge(self.cv_agents_details[2]['name'], "selector")
        graph.add_edge("llm", END)
        graph.set_entry_point("coordinator")
        self.graph = graph.compile()

    def _init_and_get_required_agents(self, state: AgentState):
        """
        Determines which agents are required to answer the user's question and generates the prompt for them.

        Args:
            state (AgentState): The current state of the environment.

        Returns:
            dict: Updated state with the required agents, their prompt, and chat history.
        """
        agents, prompt = self.coordinator.answer(state['question'],
                                                 [cv_agent['name'] for cv_agent in self.cv_agents_details])
        agent_answer = {"role": "coordinator",
                        "content": f"{self.coordinator.greetings()}"
                                   f" We need to ask the agents: {agents} about the question: {prompt}"}
        print(agent_answer)
        chat_history = [agent_answer]
        return {'agents': agents, 'agents_prompt': prompt, 'context': [], 'chat_history': chat_history}

    def _get_next_agent(self, state: AgentState):
        """
        Selects the next agent to process the user's question.

        Args:
            state (AgentState): The current state of the environment.

        Returns:
            dict: Updated state with the next agent and chat history.
        """
        agents = state['agents']
        if len(agents) > 0:
            next_agent = agents.pop(0)
        else:
            next_agent = "llm"
        agent_answer = {"role": "selector",
                        "content": "Hello I am the selector agent. I decide which CV agent should be asked next."
                                   f" Next agent is {next_agent}"}
        print(agent_answer)
        chat_history = state['chat_history']
        chat_history.append(agent_answer)
        return {"agents": agents, "next_agent": next_agent, 'chat_history': chat_history}

    def _select_agent(self, state: AgentState):
        """
        Return the next agent.

        Args:
            state (AgentState): The current state of the environment.

        Returns:
            str: The name of the next agent.
        """
        return state['next_agent']

    def _get_context_cv_agent1(self, state: AgentState):
        """
        Retrieves context from the first CV agent based on the provided prompt.

        Args:
            state (AgentState): The current state of the environment.

        Returns:
            dict: Updated state with the context from the first CV agent and chat history.
        """
        answer = self.cv_agent1.answer(state['agents_prompt'])
        context = state['context']
        context.append({'candidate': self.cv_agents_details[0]['name'], 'context': answer})
        agent_answer = {"role": "agent1", "content": f"{self.cv_agent1.greetings()} {answer}"}
        print(agent_answer)
        chat_history = state['chat_history']
        chat_history.append(agent_answer)
        return {"context": context, "chat_history": chat_history}

    def _get_context_cv_agent2(self, state: AgentState):
        """
        Retrieves context from the second CV agent based on the provided prompt.

        Args:
            state (AgentState): The current state of the environment.

        Returns:
            dict: Updated state with the context from the second CV agent and chat history.
        """
        answer = self.cv_agent2.answer(state['agents_prompt'])
        context = state['context']
        context.append({'candidate': self.cv_agents_details[1]['name'], 'context': answer})
        agent_answer = {"role": "agent2", "content": f"{self.cv_agent2.greetings()} {answer}"}
        print(agent_answer)
        chat_history = state['chat_history']
        chat_history.append(agent_answer)
        return {"context": context, "chat_history": chat_history}

    def _get_context_cv_agent3(self, state: AgentState):
        """
        Retrieves context from the third CV agent based on the provided prompt.

        Args:
            state (AgentState): The current state of the environment.

        Returns:
            dict: Updated state with the context from the third CV agent and chat history.
        """
        answer = self.cv_agent3.answer(state['agents_prompt'])
        context = state['context']
        context.append({'candidate': self.cv_agents_details[2]['name'], 'context': answer})
        agent_answer = {"role": "agent3", "content": f"{self.cv_agent3.greetings()} {answer}"}
        print(agent_answer)
        chat_history = state['chat_history']
        chat_history.append(agent_answer)
        return {"context": context, "chat_history": chat_history}

    def _answer_question(self, state: AgentState):
        """
        Generates the final answer to the user's question using the principal LLM agent.

        Args:
            state (AgentState): The current state of the environment.

        Returns:
            dict: Updated state with the final answer and chat history.
        """
        context = state['context']
        answer = self.llm.answer(state['question'], context)
        agent_answer = {"role": "llm", "content": f"{self.llm.greetings()} {answer}"}
        print(agent_answer)
        chat_history = state['chat_history']
        chat_history.append(agent_answer)
        return {"answer": answer, "chat_history": chat_history}
