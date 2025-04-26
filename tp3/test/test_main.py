import os
import sys
from unittest.mock import patch, MagicMock

from streamlit.testing.v1 import AppTest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))


def test_main_initializes_app():
    # Given
    with patch("agent.AgentEnvironment.AgentEnvironment", autospec=True) as agent_env_mock:
        agent_env_mock.return_value = MagicMock()
        app = AppTest.from_file("../src/main.py")

        # When
        app.run()

        # Then
        assert not app.exception


def test_main_user_input_calls_agent():
    # Given
    with patch("agent.AgentEnvironment.AgentEnvironment", autospec=True) as agent_env_mock:
        env_mock = MagicMock()
        agent_env_mock.return_value = env_mock
        app = AppTest.from_file("../src/main.py")

        # When
        app.run()
        app.chat_input("user_question").set_value("question").run()

        # Then
        assert not app.exception
        env_mock.graph.invoke.assert_called_once()
