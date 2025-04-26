from unittest.mock import patch, MagicMock, mock_open

from src.agent.AgentDeductions import AgentDeductions


def test_init_reads_csv_file():
    # Given
    mock_csv_data = "Category;Maximum deduction\nGeneral;100"
    with patch("builtins.open", mock_open(read_data=mock_csv_data)), \
            patch("src.agent.AgentDeductions.SingletonGroq") as singletonGroqMock:
        groq_mock = MagicMock()
        singletonGroqMock.return_value = singletonGroqMock
        singletonGroqMock.groq = groq_mock

        # When
        deductions = AgentDeductions("deductions")

        # Then
        assert deductions.deductions_data == [["Category", "Maximum deduction"], ["General", "100"]]


def test_answer_calls_chat_completion():
    # Given
    with patch("src.agent.AgentDeductions.SingletonGroq") as singletonGroqMock, \
            patch("src.agent.AgentDeductions.AgentDeductions.read_csv",
                  return_value=[["Category", "Max Amount"], ["General", "1000"]]):
        groq_mock = MagicMock()
        chat_completion = MagicMock()
        singletonGroqMock.return_value = singletonGroqMock
        singletonGroqMock.groq = groq_mock
        groq_mock.chat.completions.create.return_value = chat_completion
        deductions = AgentDeductions("deductions.csv")

        # When
        deductions.answer("test question")

        # Then
        assert groq_mock.chat.completions.create.called


def test_answer_returns_chat_completion_content_and_tokens():
    # Given
    with patch("src.agent.AgentDeductions.SingletonGroq") as singletonGroqMock, \
            patch("src.agent.AgentDeductions.AgentDeductions.read_csv",
                  return_value=[["Category", "Max Amount"], ["General", "1000"]]):
        groq_mock = MagicMock()
        chat_completion = MagicMock()
        singletonGroqMock.return_value = singletonGroqMock
        singletonGroqMock.groq = groq_mock
        expected_answer = {"thought": "some thought",
                           "deductions": "Applicable deductions: category1: 100, category2: 200"}
        groq_mock.chat.completions.create.return_value = chat_completion
        chat_completion.usage.prompt_tokens = 10
        chat_completion.usage.completion_tokens = 50
        chat_completion.choices = [MagicMock()]
        chat_completion.choices[0].message.content = str(expected_answer)
        deductions = AgentDeductions("deductions.csv")

        # When
        answer, tokens = deductions.answer("test question")

        # Then
        assert answer == expected_answer
        assert tokens == (10, 50)


def test_answer_with_invalid_format_returns_default_answer():
    # Given
    with patch("src.agent.AgentDeductions.SingletonGroq") as singletonGroqMock, \
            patch("src.agent.AgentDeductions.AgentDeductions.read_csv",
                  return_value=[["Category", "Max Amount"], ["General", "1000"]]):
        groq_mock = MagicMock()
        chat_completion = MagicMock()
        singletonGroqMock.return_value = singletonGroqMock
        singletonGroqMock.groq = groq_mock
        groq_mock.chat.completions.create.return_value = chat_completion
        chat_completion.choices = [MagicMock()]
        chat_completion.choices[0].message.content = "invalid format"
        deductions = AgentDeductions("deductions.csv")

        # When
        answer, tokens = deductions.answer("invalid input")

        # Then
        assert "I don't know" in answer["deductions"]
        assert tokens == (0, 0)


def test_answer_uses_deductions_data_in_prompt():
    # Given
    with patch("src.agent.AgentDeductions.SingletonGroq") as singletonGroqMock, \
            patch("src.agent.AgentDeductions.AgentDeductions.read_csv",
                  return_value=[["Category", "Max Amount"], ["General", "1000"]]):
        groq_mock = MagicMock()
        singletonGroqMock.return_value = singletonGroqMock
        singletonGroqMock.groq = groq_mock
        deductions = AgentDeductions("deductions.csv")

        # When
        deductions.answer("test question")

        # Then
        _, kwargs = groq_mock.chat.completions.create.call_args
        assert "General" in str(kwargs["messages"][0]["content"])
        assert "1000" in str(kwargs["messages"][0]["content"])
