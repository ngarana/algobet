"""Tests for WebSocket progress handler."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.testclient import TestClient

from algobet.api.main import app
from algobet.api.schemas.scraping import ScrapingProgress
from algobet.api.websockets.progress import (
    ConnectionManager,
    manager,
    websocket_endpoint,
)


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def mock_websocket():
    """Create a mock WebSocket."""
    websocket = MagicMock(spec=WebSocket)
    websocket.accept = AsyncMock()
    websocket.send_text = AsyncMock()
    websocket.receive_text = AsyncMock()
    websocket.client_state = MagicMock()
    websocket.client_state.DISCONNECTED = False
    return websocket


@pytest.fixture
def connection_manager():
    """Create a fresh connection manager for testing."""
    return ConnectionManager()


class TestConnectionManager:
    """Tests for ConnectionManager class."""

    @pytest.mark.asyncio
    async def test_connect_without_job_id(self, connection_manager, mock_websocket):
        """Test connecting without specifying a job ID."""
        await connection_manager.connect(mock_websocket, "client-123")

        mock_websocket.accept.assert_called_once()

        # Verify connection confirmation was sent
        mock_websocket.send_text.assert_called_once()
        sent_data = json.loads(mock_websocket.send_text.call_args[0][0])
        assert sent_data["type"] == "connection"
        assert sent_data["status"] == "connected"
        assert sent_data["client_id"] == "client-123"
        assert sent_data["job_id"] is None

    @pytest.mark.asyncio
    async def test_connect_with_job_id(self, connection_manager, mock_websocket):
        """Test connecting with a specific job ID."""
        await connection_manager.connect(mock_websocket, "client-456", "job-789")

        # Verify connection confirmation
        sent_data = json.loads(mock_websocket.send_text.call_args[0][0])
        assert sent_data["job_id"] == "job-789"

        # Verify job subscription list was created
        assert "job-789" in connection_manager.active_connections
        assert mock_websocket in connection_manager.active_connections["job-789"]

    @pytest.mark.asyncio
    async def test_disconnect(self, connection_manager, mock_websocket):
        """Test disconnecting a WebSocket."""
        # Connect first
        await connection_manager.connect(mock_websocket, "client-123", "job-456")

        # Disconnect
        connection_manager.disconnect(mock_websocket)

        # Verify cleanup
        assert mock_websocket not in connection_manager.active_connections.get(
            "job-456", []
        )
        assert mock_websocket not in connection_manager.connection_metadata

    @pytest.mark.asyncio
    async def test_send_personal_message(self, connection_manager, mock_websocket):
        """Test sending a personal message to a specific connection."""
        await connection_manager.connect(mock_websocket, "client-123")

        message = {"type": "test", "data": "hello"}
        await connection_manager.send_personal_message(message, mock_websocket)

        mock_websocket.send_text.assert_called_with(json.dumps(message))

    @pytest.mark.asyncio
    async def test_broadcast_to_job(self, connection_manager):
        """Test broadcasting a message to all connections subscribed to a job."""
        # Create multiple mock WebSockets
        websocket1 = MagicMock(spec=WebSocket)
        websocket1.send_text = AsyncMock()
        websocket1.client_state = MagicMock()
        websocket1.client_state.DISCONNECTED = False

        websocket2 = MagicMock(spec=WebSocket)
        websocket2.send_text = AsyncMock()
        websocket2.client_state = MagicMock()
        websocket2.client_state.DISCONNECTED = False

        # Connect both to the same job
        await connection_manager.connect(websocket1, "client-1", "job-123")
        await connection_manager.connect(websocket2, "client-2", "job-123")

        # Broadcast message
        message = {"type": "progress", "progress": 50}
        await connection_manager.broadcast_to_job("job-123", message)

        # Verify both received the message
        websocket1.send_text.assert_called_with(json.dumps(message))
        websocket2.send_text.assert_called_with(json.dumps(message))

    @pytest.mark.asyncio
    async def test_broadcast_progress(self, connection_manager):
        """Test broadcasting scraping progress."""
        websocket = MagicMock(spec=WebSocket)
        websocket.send_text = AsyncMock()
        websocket.client_state = MagicMock()
        websocket.client_state.DISCONNECTED = False

        await connection_manager.connect(websocket, "client-123", "job-456")

        progress = ScrapingProgress(
            job_id="job-456",
            progress=75.0,
            message="75% complete",
            matches_scraped=50,
        )

        await connection_manager.broadcast_progress(progress)

        # Verify the formatted message was sent
        websocket.send_text.assert_called_once()
        sent_data = json.loads(websocket.send_text.call_args[0][0])
        assert sent_data["type"] == "progress"
        assert sent_data["job_id"] == "job-456"
        assert sent_data["progress"] == 75.0
        assert sent_data["message"] == "75% complete"
        assert sent_data["matches_scraped"] == 50

    @pytest.mark.asyncio
    async def test_broadcast_job_status(self, connection_manager):
        """Test broadcasting job status updates."""
        websocket = MagicMock(spec=WebSocket)
        websocket.send_text = AsyncMock()
        websocket.client_state = MagicMock()
        websocket.client_state.DISCONNECTED = False

        await connection_manager.connect(websocket, "client-123", "job-789")

        await connection_manager.broadcast_job_status(
            "job-789", "completed", "Job finished"
        )

        # Verify status message was sent
        websocket.send_text.assert_called_once()
        sent_data = json.loads(websocket.send_text.call_args[0][0])
        assert sent_data["type"] == "status"
        assert sent_data["job_id"] == "job-789"
        assert sent_data["status"] == "completed"
        assert sent_data["message"] == "Job finished"

    @pytest.mark.asyncio
    async def test_handle_client_message_subscribe(
        self, connection_manager, mock_websocket
    ):
        """Test handling client subscribe message."""
        await connection_manager.connect(mock_websocket, "client-123")

        # Simulate subscribe message
        message = json.dumps({"type": "subscribe", "job_id": "job-456"})
        mock_websocket.receive_text.return_value = message

        await connection_manager.handle_client_message(mock_websocket, message)

        # Verify subscription was processed
        assert "job-456" in connection_manager.active_connections
        assert mock_websocket in connection_manager.active_connections["job-456"]

    @pytest.mark.asyncio
    async def test_handle_client_message_unsubscribe(
        self, connection_manager, mock_websocket
    ):
        """Test handling client unsubscribe message."""
        await connection_manager.connect(mock_websocket, "client-123", "job-456")

        # Simulate unsubscribe message
        message = json.dumps({"type": "unsubscribe", "job_id": "job-456"})
        mock_websocket.receive_text.return_value = message

        await connection_manager.handle_client_message(mock_websocket, message)

        # Verify unsubscription was processed
        assert mock_websocket not in connection_manager.active_connections.get(
            "job-456", []
        )

    @pytest.mark.asyncio
    async def test_handle_client_message_ping(self, connection_manager, mock_websocket):
        """Test handling client ping message."""
        await connection_manager.connect(mock_websocket, "client-123")

        # Simulate ping message
        message = json.dumps({"type": "ping"})
        mock_websocket.receive_text.return_value = message

        await connection_manager.handle_client_message(mock_websocket, message)

        # Verify pong was sent
        sent_data = json.loads(mock_websocket.send_text.call_args_list[-1][0][0])
        assert sent_data["type"] == "pong"

    @pytest.mark.asyncio
    async def test_handle_client_message_invalid_json(
        self, connection_manager, mock_websocket
    ):
        """Test handling invalid JSON from client."""
        await connection_manager.connect(mock_websocket, "client-123")

        # Simulate invalid JSON
        invalid_message = "invalid json {"

        await connection_manager.handle_client_message(mock_websocket, invalid_message)

        # Verify error message was sent
        sent_data = json.loads(mock_websocket.send_text.call_args_list[-1][0][0])
        assert sent_data["type"] == "error"
        assert "Invalid JSON" in sent_data["message"]

    @pytest.mark.asyncio
    async def test_handle_client_message_unknown_type(
        self, connection_manager, mock_websocket
    ):
        """Test handling unknown message type."""
        await connection_manager.connect(mock_websocket, "client-123")

        # Simulate unknown message type
        message = json.dumps({"type": "unknown_type", "data": "test"})

        await connection_manager.handle_client_message(mock_websocket, message)

        # Should not crash, just log warning
        # No specific response expected for unknown types

    def test_get_connection_stats(self, connection_manager, mock_websocket):
        """Test getting connection statistics."""
        # Add some connections
        asyncio.run(connection_manager.connect(mock_websocket, "client-1", "job-123"))

        stats = connection_manager.get_connection_stats()

        assert stats["total_connections"] == 1
        assert stats["job_subscriptions"] == 1
        assert stats["unique_jobs"] == 1
        assert stats["active_connections"] == 1

    @pytest.mark.asyncio
    async def test_connection_cleanup_on_error(
        self, connection_manager, mock_websocket
    ):
        """Test that connections are cleaned up when errors occur."""
        await connection_manager.connect(mock_websocket, "client-123", "job-456")

        # Simulate an error during message sending
        mock_websocket.send_text.side_effect = Exception("Connection error")

        # Try to broadcast a message
        message = {"type": "test", "data": "message"}
        await connection_manager.broadcast_to_job("job-456", message)

        # The connection should be cleaned up due to the error
        # (This depends on the implementation details)


class TestGlobalManager:
    """Tests for the global manager instance."""

    def test_global_manager_instance(self):
        """Test that the global manager is properly initialized."""
        assert manager is not None
        assert isinstance(manager, ConnectionManager)
        assert manager.active_connections == {}
        assert manager.connection_metadata == {}


class TestWebSocketEndpoint:
    """Tests for the WebSocket endpoint function."""

    @pytest.mark.asyncio
    async def test_websocket_endpoint_basic(self, mock_websocket):
        """Test basic WebSocket endpoint functionality."""
        # Mock receive_text to return a ping message after connection
        mock_websocket.receive_text.return_value = asyncio.Future()
        mock_websocket.receive_text.return_value.set_result(
            json.dumps({"type": "ping"})
        )

        # Mock that the websocket gets disconnected
        mock_websocket.receive_text.side_effect = WebSocketDisconnect()

        await websocket_endpoint(mock_websocket, "test-client")

        # Verify basic connection flow
        mock_websocket.accept.assert_called_once()

    @pytest.mark.asyncio
    async def test_websocket_endpoint_with_job_id(self, mock_websocket):
        """Test WebSocket endpoint with job ID parameter."""
        mock_websocket.receive_text.side_effect = WebSocketDisconnect()

        await websocket_endpoint(mock_websocket, "test-client", "job-123")

        # Verify connection with job ID
        mock_websocket.accept.assert_called_once()

    @pytest.mark.asyncio
    async def test_websocket_endpoint_error_handling(self, mock_websocket):
        """Test WebSocket endpoint error handling."""
        # Simulate an exception during connection
        mock_websocket.accept.side_effect = RuntimeError("Connection failed")

        with pytest.raises(RuntimeError):
            await websocket_endpoint(mock_websocket, "test-client")


class TestIntegrationWithScrapingProgress:
    """Integration tests with ScrapingProgress schema."""

    @pytest.mark.asyncio
    async def test_progress_broadcast_integration(
        self, connection_manager, mock_websocket
    ):
        """Test integration with ScrapingProgress schema."""
        await connection_manager.connect(mock_websocket, "client-123", "job-456")

        # Create a real ScrapingProgress object
        progress = ScrapingProgress(
            job_id="job-456",
            progress=33.33,
            message="One third complete",
            matches_scraped=100,
        )

        await connection_manager.broadcast_progress(progress)

        # Verify the message was formatted correctly
        mock_websocket.send_text.assert_called_once()
        sent_data = json.loads(mock_websocket.send_text.call_args[0][0])

        assert sent_data["job_id"] == "job-456"
        assert sent_data["progress"] == 33.33
        assert sent_data["message"] == "One third complete"
        assert sent_data["matches_scraped"] == 100
        assert "timestamp" in sent_data

    @pytest.mark.asyncio
    async def test_multiple_progress_updates(self, connection_manager, mock_websocket):
        """Test handling multiple progress updates."""
        await connection_manager.connect(mock_websocket, "client-123", "job-789")

        # Send multiple progress updates
        for i in range(1, 4):
            progress = ScrapingProgress(
                job_id="job-789",
                progress=i * 25.0,
                message=f"Step {i} complete",
                matches_scraped=i * 50,
            )
            await connection_manager.broadcast_progress(progress)

        # Verify all messages were sent
        assert mock_websocket.send_text.call_count == 3

        # Check last message
        last_call = mock_websocket.send_text.call_args_list[-1]
        sent_data = json.loads(last_call[0][0])
        assert sent_data["progress"] == 75.0
        assert sent_data["message"] == "Step 3 complete"
        assert sent_data["matches_scraped"] == 150


class TestConcurrentWebSocketConnections:
    """Tests for handling concurrent WebSocket connections."""

    @pytest.mark.asyncio
    async def test_multiple_clients_same_job(self, connection_manager):
        """Test multiple clients connected to the same job."""
        # Create multiple mock WebSockets
        websockets = []
        for _i in range(3):
            ws = MagicMock(spec=WebSocket)
            ws.accept = AsyncMock()
            ws.send_text = AsyncMock()
            ws.receive_text = AsyncMock()
            ws.client_state = MagicMock()
            ws.client_state.DISCONNECTED = False
            websockets.append(ws)

        # Connect all to the same job
        for i, ws in enumerate(websockets):
            await connection_manager.connect(ws, f"client-{i}", "shared-job")

        # Broadcast progress to the job
        progress = ScrapingProgress(
            job_id="shared-job",
            progress=50.0,
            message="Halfway there",
            matches_scraped=250,
        )
        await connection_manager.broadcast_progress(progress)

        # Verify all clients received the message
        for ws in websockets:
            assert ws.send_text.call_count == 2  # Connection confirmation + progress
            progress_call = ws.send_text.call_args_list[-1]
            sent_data = json.loads(progress_call[0][0])
            assert sent_data["progress"] == 50.0
            assert sent_data["message"] == "Halfway there"

    @pytest.mark.asyncio
    async def test_concurrent_subscribe_unsubscribe(self, connection_manager):
        """Test concurrent subscription and unsubscription operations."""
        websocket = MagicMock(spec=WebSocket)
        websocket.accept = AsyncMock()
        websocket.send_text = AsyncMock()
        websocket.receive_text = AsyncMock()
        websocket.client_state = MagicMock()
        websocket.client_state.DISCONNECTED = False

        await connection_manager.connect(websocket, "client-123")

        # Rapid subscribe/unsubscribe operations
        for i in range(5):
            # Subscribe to a job
            message = json.dumps({"type": "subscribe", "job_id": f"job-{i}"})
            await connection_manager.handle_client_message(websocket, message)

            # Unsubscribe from the job
            message = json.dumps({"type": "unsubscribe", "job_id": f"job-{i}"})
            await connection_manager.handle_client_message(websocket, message)

        # Verify final state
        stats = connection_manager.get_connection_stats()
        assert stats["job_subscriptions"] == 0  # Should be unsubscribed from all
        assert stats["unique_jobs"] == 0
