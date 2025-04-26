from unittest.mock import patch, MagicMock
from src.agent.client.SingletonGroq import SingletonGroq


def test_get_instance_return_same_instance():
    # Given
    with patch("src.agent.client.SingletonGroq.Groq", MagicMock()):
        # When
        instance1 = SingletonGroq()
        instance2 = SingletonGroq()

        # Then
        assert instance2 is instance1