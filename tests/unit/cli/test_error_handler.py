"""Unit tests for error handler module."""

import pytest
from unittest.mock import patch, MagicMock
from click.testing import CliRunner
import click
from algobet.cli.error_handler import handle_errors
from algobet.exceptions import (
    AlgoBetError,
    DatabaseConnectionError,
    ValidationError,
    ConfigurationError,
    ScrapingError,
    PredictionError
)


class TestErrorHandler:
    """Test cases for the error handler functionality."""
    
    def test_handle_errors_decorator_success(self):
        """Test handle_errors decorator with a successful function."""
        @handle_errors
        def successful_func():
            return "success"
        
        result = successful_func()
        assert result == "success"
    
    @patch('sys.exit')
    def test_handle_errors_decorator_algotbet_exception(self, mock_exit):
        """Test handle_errors decorator with an AlgoBetException."""
        @handle_errors
        def raising_func():
            raise ValidationError("Test validation error")
        
        # The function should call sys.exit but not raise an exception
        raising_func()
        mock_exit.assert_called_once_with(80)  # ValidationError exit code
    
    @patch('sys.exit')
    def test_handle_errors_decorator_general_exception(self, mock_exit):
        """Test handle_errors decorator with a general exception."""
        @handle_errors
        def raising_func():
            raise ValueError("General error")
        
        # The function should call sys.exit but not raise an exception
        raising_func()
        mock_exit.assert_called_once_with(1)  # General error exit code
    
    @patch('sys.exit')
    def test_handle_errors_decorator_database_error(self, mock_exit):
        """Test handle_errors decorator with a DatabaseConnectionError."""
        @handle_errors
        def raising_func():
            raise DatabaseConnectionError("DB connection failed")
        
        raising_func()
        mock_exit.assert_called_once_with(11)  # DatabaseConnectionError exit code
    
    @patch('sys.exit')
    def test_handle_errors_decorator_scraping_error(self, mock_exit):
        """Test handle_errors decorator with a ScrapingError."""
        @handle_errors
        def raising_func():
            raise ScrapingError("Scraping failed")
        
        raising_func()
        mock_exit.assert_called_once_with(40)  # ScrapingError exit code
    
    @patch('sys.exit')
    def test_handle_errors_decorator_prediction_error(self, mock_exit):
        """Test handle_errors decorator with a PredictionError."""
        @handle_errors
        def raising_func():
            raise PredictionError("Prediction failed")
        
        raising_func()
        mock_exit.assert_called_once_with(60)  # PredictionError exit code


@patch('sys.exit')
def test_handle_errors_with_debug_mode(mock_exit):
    """Test that debug mode shows full tracebacks."""
    @handle_errors
    def raising_func():
        raise ValidationError("Debug test error")
    
    # Mock the config to enable debug mode
    with patch('algobet.cli.error_handler.get_config') as mock_get_config:
        mock_config = MagicMock()
        mock_config.cli.debug = True  # Note: it's config.cli.debug, not config.app.debug
        mock_get_config.return_value = mock_config
        
        # In debug mode, it should still call sys.exit but with more info logged
        raising_func()
        # In debug mode, it should still call sys.exit with the exception's exit code
        mock_exit.assert_called_once_with(80)  # ValidationError exit code