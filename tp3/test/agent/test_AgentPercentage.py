from unittest.mock import patch, MagicMock, mock_open

from src.agent.AgentPercentage import AgentPercentage


def test_init_reads_csv_file():
    # Given
    mock_csv_data = "Range;Fixed Amount;Base Surplus;Percentage\n0-1000;100;500;10%"
    with patch("builtins.open", mock_open(read_data=mock_csv_data)), \
         patch("src.agent.AgentPercentage.SingletonGroq") as singletonGroqMock:
        groq_mock = MagicMock()
        singletonGroqMock.return_value = singletonGroqMock
        singletonGroqMock.groq = groq_mock

        # When
        percentage = AgentPercentage("tax.csv")

        # Then
        assert percentage.tax_data == [["Range", "Fixed Amount", "Base Surplus", "Percentage"],
                                       ["0-1000", "100", "500", "10%"]]


def test_answer_calls_chat_completion():
    # Given
    with patch("src.agent.AgentPercentage.SingletonGroq") as singletonGroqMock, \
            patch("src.agent.AgentPercentage.AgentPercentage.read_csv",
                  return_value=[["Range", "Fixed Amount", "Base Surplus", "Percentage"]]):
        groq_mock = MagicMock()
        chat_completion = MagicMock()
        singletonGroqMock.return_value = singletonGroqMock
        singletonGroqMock.groq = groq_mock
        groq_mock.chat.completions.create.return_value = chat_completion
        percentage = AgentPercentage("tax.csv")

        # When
        percentage.answer("test question")

        # Then
        assert groq_mock.chat.completions.create.called


def test_answer_returns_chat_completion_content_and_tokens():
    # Given
    with patch("src.agent.AgentPercentage.SingletonGroq") as singletonGroqMock, \
            patch("src.agent.AgentPercentage.AgentPercentage.read_csv",
                  return_value=[["Range", "Fixed Amount", "Base Surplus", "Percentage"]]):
        groq_mock = MagicMock()
        chat_completion = MagicMock()
        singletonGroqMock.return_value = singletonGroqMock
        singletonGroqMock.groq = groq_mock
        expected_answer = {"thought": "some thought", "percentage": "Fixed: 100, Base: 500, Percent: 10%"}
        groq_mock.chat.completions.create.return_value = chat_completion
        chat_completion.usage.prompt_tokens = 15
        chat_completion.usage.completion_tokens = 30
        chat_completion.choices = [MagicMock()]
        chat_completion.choices[0].message.content = str(expected_answer)
        percentage = AgentPercentage("tax.csv")

        # When
        answer, tokens = percentage.answer("test question")

        # Then
        assert answer == expected_answer
        assert tokens == (15, 30)


def test_answer_with_invalid_format_returns_default_answer():
    # Given
    with patch("src.agent.AgentPercentage.SingletonGroq") as singletonGroqMock, \
            patch("src.agent.AgentPercentage.AgentPercentage.read_csv",
                  return_value=[["Range", "Fixed Amount", "Base Surplus", "Percentage"]]):
        groq_mock = MagicMock()
        chat_completion = MagicMock()
        singletonGroqMock.return_value = singletonGroqMock
        singletonGroqMock.groq = groq_mock
        groq_mock.chat.completions.create.return_value = chat_completion
        chat_completion.choices = [MagicMock()]
        chat_completion.choices[0].message.content = "invalid format"
        percentage = AgentPercentage("tax.csv")

        # When
        answer, tokens = percentage.answer("invalid input")

        # Then
        assert "I don't know" in answer["percentage"]
        assert tokens == (0, 0)


def test_answer_uses_tax_data_in_prompt():
    # Given
    with patch("src.agent.AgentPercentage.SingletonGroq") as singletonGroqMock, \
            patch("src.agent.AgentPercentage.AgentPercentage.read_csv",
                  return_value=[["Range", "Fixed Amount", "Base Surplus", "Percentage"],
                                ["0-1000", "100", "500", "10%"]]):
        groq_mock = MagicMock()
        singletonGroqMock.return_value = singletonGroqMock
        singletonGroqMock.groq = groq_mock
        percentage = AgentPercentage("tax.csv")

        # When
        percentage.answer("test question")

        # Then
        _, kwargs = groq_mock.chat.completions.create.call_args
        assert "0-1000" in str(kwargs["messages"][0]["content"])
        assert "100" in str(kwargs["messages"][0]["content"])
        assert "10%" in str(kwargs["messages"][0]["content"])
