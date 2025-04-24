from typing import TypedDict

from langgraph.graph import StateGraph

from AgentAccountant import AgentAccountant
from AgentCalculator import AgentCalculator
from AgentDeductions import AgentDeductions
from AgentPercentage import AgentPercentage


class AgentState(TypedDict):
    question: str
    chat_history: list[dict[str, str]]
    answer: str
    next_agent: str
    agent_prompt: str


class AgentEnvironment:
    """
    This class manages the environment for coordinating multiple agents to answer user questions based on CV data.
    It initializes the agents, sets up a state graph, and orchestrates the flow of information between agents.
    """

    def __init__(self):
        """
        Initializes the AgentEnvironment with the provided CV files, sets up agents, and compiles the state graph.
        """
        # Initialize the agents
        self.accountant = AgentAccountant()
        self.calculator = AgentCalculator()
        self.deductions = AgentDeductions("resources/deductions_data.csv")
        self.percentage = AgentPercentage("resources/tax_data.csv")

        # Initialize the state graph
        graph = StateGraph(AgentState)
        graph.add_node("accountant", self.process_accountant)
        graph.add_node("calculator", self.process_calculator)
        graph.add_node("deductions", self.process_deductions)
        graph.add_node("percentage", self.process_percentage)
        graph.add_conditional_edges("accountant", self._select_agent)
        graph.add_edge("calculator", "accountant")
        graph.add_edge("deductions", "accountant")
        graph.add_edge("percentage", "accountant")
        graph.set_entry_point("accountant")
        self.graph = graph.compile()

    def process_accountant(self, state: AgentState):
        """
        Determines which agents are required to answer the user's question and generates the prompt for them.

        Args:
            state (AgentState): The current state of the environment.

        Returns:
            dict: Updated state with the required agents, their prompt, and chat history.
        """
        if 'chat_history' in state:
            chat_history = state['chat_history']
        else:
            chat_history = []
        accountant_answer = self.accountant.answer(state['question'], chat_history)
        print(accountant_answer)
        chat_history.append({"role": "assistant", "content": str(accountant_answer)})
        agent_prompt = None
        final_answer = None
        if 'deductions' in accountant_answer:
            next_agent = "deductions"
            agent_prompt = accountant_answer['deductions']
        elif 'calculator' in accountant_answer:
            next_agent = "calculator"
            agent_prompt = accountant_answer['calculator']
        elif 'percentage' in accountant_answer:
            next_agent = "percentage"
            agent_prompt = accountant_answer['percentage']
        elif 'answer' in accountant_answer:
            next_agent = "END"
            final_answer = accountant_answer['answer']
        else:
            next_agent = "END"
            final_answer = "I don't know"
        return {'next_agent': next_agent, 'agent_prompt': agent_prompt, 'answer': final_answer, 'chat_history': chat_history}

    def _select_agent(self, state: AgentState):
        """
        Return the next agent.

        Args:
            state (AgentState): The current state of the environment.

        Returns:
            str: The name of the next agent.
        """
        return state['next_agent']

    def process_calculator(self, state: AgentState):
        chat_history = state['chat_history']
        calculator_answer = self.calculator.answer(state['agent_prompt'])
        print(calculator_answer)
        chat_history.append({"role": "user", "content": str(calculator_answer)})
        return {"chat_history": chat_history}

    def process_percentage(self, state: AgentState):
        chat_history = state['chat_history']
        percentage_answer = self.percentage.answer(state['agent_prompt'])
        print(percentage_answer)
        chat_history.append({"role": "user", "content": str(percentage_answer)})
        return {"chat_history": chat_history}

    def process_deductions(self, state: AgentState):
        chat_history = state['chat_history']
        deductions_answer = self.deductions.answer(state['agent_prompt'])
        print(deductions_answer)
        chat_history.append({"role": "user", "content": str(deductions_answer)})
        return {"chat_history": chat_history}


abot = AgentEnvironment()
question= "How much do I have to pay in taxes next month? My deductions are: house=$1000, expenses=$5000, fridge=$2000. My month income is $1000"
answer = abot.graph.invoke({"question": question})