from unittest.mock import patch, MagicMock

from src.agent.AgentAccountant import AgentAccountant


def test_answer_calls_chat_completion():
    # Given
    with patch("src.agent.AgentAccountant.SingletonGroq") as singletonGroqMock:
        groq_mock = MagicMock()
        chat_completion = MagicMock()
        singletonGroqMock.return_value = singletonGroqMock
        singletonGroqMock.groq = groq_mock
        groq_mock.chat.completions.create.return_value = chat_completion
        accountant = AgentAccountant()

        # When
        accountant.answer("test question")

        # Then
        assert groq_mock.chat.completions.create.called


def test_answer_returns_chat_completion_content_and_tokens():
    # Given
    with patch("src.agent.AgentAccountant.SingletonGroq") as singletonGroqMock:
        groq_mock = MagicMock()
        chat_completion = MagicMock()
        singletonGroqMock.return_value = singletonGroqMock
        singletonGroqMock.groq = groq_mock
        expected_answer = {"thought": "some thought", "answer": "some answer"}
        groq_mock.chat.completions.create.return_value = chat_completion
        chat_completion.usage.prompt_tokens = 10
        chat_completion.usage.completion_tokens = 50
        chat_completion.choices = [MagicMock()]
        chat_completion.choices[0].message.content = str(expected_answer)
        accountant = AgentAccountant()

        # When
        answer, tokens = accountant.answer("test question")

        # Then
        assert answer == expected_answer
        assert tokens == (10, 50)


def test_answer_no_reasoning_uses_user_question():
    # Given
    with patch("src.agent.AgentAccountant.SingletonGroq") as singletonGroqMock:
        groq_mock = MagicMock()
        singletonGroqMock.return_value = singletonGroqMock
        singletonGroqMock.groq = groq_mock
        user_input = "test question"
        accountant = AgentAccountant()

        # When
        accountant.answer(user_input)

        # Then
        _, kwargs = groq_mock.chat.completions.create.call_args
        assert user_input in str(kwargs["messages"][1]["content"])


def test_answer_with_reasoning_uses_reasoning():
    # Given
    with patch("src.agent.AgentAccountant.SingletonGroq") as singletonGroqMock:
        groq_mock = MagicMock()
        singletonGroqMock.return_value = singletonGroqMock
        singletonGroqMock.groq = groq_mock
        user_input = "test question"
        reasoning = "some reasoning"
        accountant = AgentAccountant()

        # When
        accountant.answer(user_input, [reasoning])

        # Then
        _, kwargs = groq_mock.chat.completions.create.call_args
        assert user_input in str(kwargs["messages"][1]["content"])
        assert reasoning in str(kwargs["messages"][2])


def test_answer_completion_with_incorrect_format_returns_default_answer():
    # Given
    with patch("src.agent.AgentAccountant.SingletonGroq") as singletonGroqMock:
        groq_mock = MagicMock()
        chat_completion = MagicMock()
        singletonGroqMock.return_value = singletonGroqMock
        singletonGroqMock.groq = groq_mock
        groq_mock.chat.completions.create.return_value = chat_completion
        chat_completion.choices = [MagicMock()]
        chat_completion.choices[0].message.content = "incorrect format"
        accountant = AgentAccountant()

        # When
        answer, tokens = accountant.answer("test input")

        # Then
        assert "don't know" in str(answer)
        assert tokens == (0, 0)
