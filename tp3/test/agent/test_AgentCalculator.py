from unittest.mock import patch, MagicMock
from src.agent.AgentCalculator import AgentCalculator


def test_answer_calls_chat_completion():
    # Given
    with patch("src.agent.AgentCalculator.SingletonGroq") as singletonGroqMock:
        groq_mock = MagicMock()
        chat_completion = MagicMock()
        singletonGroqMock.return_value = singletonGroqMock
        singletonGroqMock.groq = groq_mock
        groq_mock.chat.completions.create.return_value = chat_completion
        calculator = AgentCalculator()

        # When
        calculator.answer("2 + 2")

        # Then
        assert groq_mock.chat.completions.create.called


def test_answer_returns_chat_completion_content_and_tokens():
    # Given
    with patch("src.agent.AgentCalculator.SingletonGroq") as singletonGroqMock:
        groq_mock = MagicMock()
        chat_completion = MagicMock()
        singletonGroqMock.return_value = singletonGroqMock
        singletonGroqMock.groq = groq_mock
        groq_mock.chat.completions.create.return_value = chat_completion
        chat_completion.usage.prompt_tokens = 5
        chat_completion.usage.completion_tokens = 10
        chat_completion.choices = [MagicMock()]
        chat_completion.choices[0].message.content = "{'thought': 'simple addition', 'calculator': '4'}"
        calculator = AgentCalculator()

        # When
        answer, tokens = calculator.answer("2 + 2")

        # Then
        assert answer == {"thought": "simple addition", "calculator": 4}
        assert tokens == (5, 10)


def test_answer_with_invalid_format_returns_default_answer():
    # Given
    with patch("src.agent.AgentCalculator.SingletonGroq") as singletonGroqMock:
        groq_mock = MagicMock()
        chat_completion = MagicMock()
        singletonGroqMock.return_value = singletonGroqMock
        singletonGroqMock.groq = groq_mock
        groq_mock.chat.completions.create.return_value = chat_completion
        chat_completion.choices = [MagicMock()]
        chat_completion.choices[0].message.content = "invalid format"
        calculator = AgentCalculator()

        # When
        answer, tokens = calculator.answer("invalid input")

        # Then
        assert "I don't know" in answer["calculator"]
        assert tokens == (0, 0)


def test_answer_evaluates_calculation_correctly():
    # Given
    with patch("src.agent.AgentCalculator.SingletonGroq") as singletonGroqMock:
        groq_mock = MagicMock()
        chat_completion = MagicMock()
        singletonGroqMock.return_value = singletonGroqMock
        singletonGroqMock.groq = groq_mock
        expected_answer = {"thought": "multiplication", "calculator": "2 * 3"}
        groq_mock.chat.completions.create.return_value = chat_completion
        chat_completion.choices = [MagicMock()]
        chat_completion.choices[0].message.content = str(expected_answer)
        calculator = AgentCalculator()

        # When
        answer, _ = calculator.answer("2 * 3")

        # Then
        assert answer["calculator"] == 6