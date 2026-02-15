"""Unit tests for logger module."""

import pytest
import logging
from unittest.mock import patch, MagicMock
from algobet.cli.logger import get_cli_logger, get_logger


class TestLogger:
    """Test cases for the CLI logger functionality."""
    
    def test_get_cli_logger(self):
        """Test getting a CLI logger instance."""
        logger = get_cli_logger("test_module")
        
        assert logger is not None
        assert "algobet.cli.test_module" in logger.name
    
    def test_get_logger(self):
        """Test getting a logger instance."""
        logger = get_logger("test_module")
        
        assert logger is not None
        assert logger.name == "algobet.test_module"
    
    def test_logger_has_echo_handler(self):
        """Test that CLI logger has EchoHandler."""
        logger = get_cli_logger("handler_test")
        
        # Verify logger has EchoHandler
        has_echo_handler = any(
            hasattr(handler, '__class__') and handler.__class__.__name__ == 'EchoHandler' 
            for handler in logger.handlers
        )
        assert has_echo_handler is True