from unittest.mock import patch

from src.agent.AgentEnvironment import AgentEnvironment, AgentState


def set_up():
    accountant_mock = patch("src.agent.AgentEnvironment.AgentAccountant").start()
    calculator_mock = patch("src.agent.AgentEnvironment.AgentCalculator").start()
    deductions_mock = patch("src.agent.AgentEnvironment.AgentDeductions").start()
    percentage_mock = patch("src.agent.AgentEnvironment.AgentPercentage").start()
    accountant_mock.return_value = accountant_mock
    calculator_mock.return_value = calculator_mock
    deductions_mock.return_value = deductions_mock
    percentage_mock.return_value = percentage_mock
    env = AgentEnvironment()
    return accountant_mock, calculator_mock, deductions_mock, percentage_mock, env


def get_agent_state(**overrides):
    state = AgentState(
        question="",
        reasoning=[],
        chat_history=[],
        answer="",
        next_agent="",
        agent_prompt="",
        token_usage_history=[],
        input_tokens=0,
        reasoning_tokens=0,
        output_tokens=0
    )
    state.update(overrides)
    return state


def test_process_accountant_first_call_calls_accountant_answer():
    # Given
    accountant_mock, _, _, _, env = set_up()
    accountant_mock.answer.return_value = ({"deductions": "test deductions"}, (10, 20))
    state = {"question": "test question"}
    env = AgentEnvironment()

    # When
    env.process_accountant(state)

    # Then
    assert accountant_mock.answer.called


def test_process_accountant_second_call_calls_accountant_answer():
    # Given
    accountant_mock, _, _, _, env = set_up()
    accountant_mock.answer.return_value = ({"deductions": "test deductions"}, (10, 20))
    state = get_agent_state(question="test question", reasoning=[{"role": "user", "content": "calculator=5"}],
                            chat_history=[{"role": "calculator", "content": "test calcs is 5"}],
                            token_usage_history=[(5, 5)])
    env = AgentEnvironment()

    # When
    result = env.process_accountant(state)

    # Then
    assert result["next_agent"] == "deductions"
    assert result["agent_prompt"] == "test deductions"
    assert accountant_mock.answer.called


def test_process_accountant_calls_deductions():
    # Given
    accountant_mock, _, _, _, env = set_up()
    accountant_mock.answer.return_value = ({"deductions": "test deductions"}, (10, 20))
    state = get_agent_state(question="test")
    env = AgentEnvironment()

    # When
    result = env.process_accountant(state)

    # Then
    assert result["next_agent"] == "deductions"
    assert result["agent_prompt"] == "test deductions"


def test_process_accountant_calls_calculator():
    # Given
    accountant_mock, _, _, _, env = set_up()
    accountant_mock.answer.return_value = ({"calculator": "test calculator"}, (10, 20))
    state = get_agent_state(question="test")
    env = AgentEnvironment()

    # When
    result = env.process_accountant(state)

    # Then
    assert result["next_agent"] == "calculator"
    assert result["agent_prompt"] == "test calculator"


def test_process_accountant_calls_percentage():
    # Given
    accountant_mock, _, _, _, env = set_up()
    accountant_mock.answer.return_value = ({"percentage": "test percentage"}, (10, 20))
    state = get_agent_state(question="test")
    env = AgentEnvironment()

    # When
    result = env.process_accountant(state)

    # Then
    assert result["next_agent"] == "percentage"
    assert result["agent_prompt"] == "test percentage"


def test_process_accountant_with_answer_calls_pricer():
    # Given
    accountant_mock, _, _, _, env = set_up()
    accountant_mock.answer.return_value = ({"answer": "final answer"}, (10, 20))
    state = get_agent_state(question="test")
    env = AgentEnvironment()

    # When
    result = env.process_accountant(state)

    # Then
    assert result["next_agent"] == "pricer"
    assert result["answer"] == "final answer"


def test_process_accountant_with_incorrect_answer_calls_pricer():
    # Given
    accountant_mock, _, _, _, env = set_up()
    accountant_mock.answer.return_value = ("Incorrect", (0, 0))
    state = get_agent_state(question="test")
    env = AgentEnvironment()

    # When
    result = env.process_accountant(state)

    # Then
    assert result["next_agent"] == "pricer"
    assert result["answer"] == "Incorrect"


def test_select_agent_returns_next_agent():
    # Given
    _, _, _, _, env = set_up()
    state = get_agent_state(next_agent="calculator")
    env = AgentEnvironment()

    # When
    result = env._select_agent(state)

    # Then
    assert result == "calculator"


def test_process_calculator_calls_calculator_answer():
    # Given
    _, calculator_mock, _, _, env = set_up()
    calculator_mock.answer.return_value = ({"calculator": "4"}, (5, 10))
    state = get_agent_state(agent_prompt="calculator:2 + 2")
    env = AgentEnvironment()

    # When
    result = env.process_calculator(state)

    # Then
    assert "4" in result["reasoning"][-1]["content"]
    assert calculator_mock.answer.called


def test_process_deductions_calls_deductions_answer():
    # Given
    _, _, deductions_mock, _, env = set_up()
    deductions_mock.answer.return_value = ({"deductions": "test deductions"}, (10, 20))
    state = get_agent_state(agent_prompt="test deductions")
    env = AgentEnvironment()

    # When
    result = env.process_deductions(state)

    # Then
    assert "test deductions" in result["reasoning"][-1]["content"]
    assert deductions_mock.answer.called


def test_process_percentage_calls_percentage_answer():
    # Given
    _, _, _, percentage_mock, env = set_up()
    percentage_mock.answer.return_value = ({"percentage": "10%"}, (15, 30))
    state = get_agent_state(agent_prompt="10%")
    env = AgentEnvironment()

    # When
    result = env.process_percentage(state)

    # Then
    assert "10%" in result["reasoning"][-1]["content"]
    assert percentage_mock.answer.called


def test_pricer_calculates_token_usage():
    # Given
    _, _, _, _, env = set_up()
    state = get_agent_state(token_usage_history=[(10, 20), (5, 10), (15, 30)])

    # When
    result = env.pricer(state)

    # Then
    assert result["input_tokens"] == 10
    assert result["reasoning_tokens"] == 50
    assert result["output_tokens"] == 30
