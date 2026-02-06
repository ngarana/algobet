"""WebSocket connection manager for real-time progress updates."""

import json
import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from algobet.api.schemas.scraping import ScrapingProgress

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections for real-time progress updates."""

    def __init__(self) -> None:
        """Initialize connection manager."""
        self.active_connections: dict[str, list[WebSocket]] = {}
        self.connection_metadata: dict[WebSocket, dict[str, Any]] = {}

    async def connect(
        self, websocket: WebSocket, client_id: str, job_id: str | None = None
    ) -> None:
        """Accept a new WebSocket connection.

        Args:
            websocket: The WebSocket connection
            client_id: Unique identifier for the client
            job_id: Optional job ID to subscribe to specific updates
        """
        await websocket.accept()

        # Store connection metadata
        self.connection_metadata[websocket] = {
            "client_id": client_id,
            "job_id": job_id,
            "connected_at": datetime.now(timezone.utc),
        }

        # Add to job-specific connection list if job_id provided
        if job_id:
            if job_id not in self.active_connections:
                self.active_connections[job_id] = []
            self.active_connections[job_id].append(websocket)

        logger.info(f"WebSocket connected: client_id={client_id}, job_id={job_id}")

        # Send connection confirmation
        await self.send_personal_message(
            {
                "type": "connection",
                "status": "connected",
                "client_id": client_id,
                "job_id": job_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            websocket,
        )

    def disconnect(self, websocket: WebSocket) -> None:
        """Remove a WebSocket connection.

        Args:
            websocket: The WebSocket connection to remove
        """
        metadata = self.connection_metadata.pop(websocket, {})
        client_id = metadata.get("client_id", "unknown")
        job_id = metadata.get("job_id")

        # Remove from job-specific connections
        if job_id and job_id in self.active_connections:
            if websocket in self.active_connections[job_id]:
                self.active_connections[job_id].remove(websocket)

            # Clean up empty connection lists
            if not self.active_connections[job_id]:
                del self.active_connections[job_id]

        logger.info(f"WebSocket disconnected: client_id={client_id}, job_id={job_id}")

    async def send_personal_message(
        self, message: dict[str, Any], websocket: WebSocket
    ) -> None:
        """Send a message to a specific connection.

        Args:
            message: Message data to send
            websocket: Target WebSocket connection
        """
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")
            self.disconnect(websocket)

    async def broadcast_to_job(self, job_id: str, message: dict[str, Any]) -> None:
        """Broadcast a message to all connections subscribed to a specific job.

        Args:
            job_id: Job identifier
            message: Message data to broadcast
        """
        if job_id not in self.active_connections:
            return

        disconnected_connections = []

        for websocket in self.active_connections[job_id]:
            try:
                await websocket.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error broadcasting to job {job_id}: {e}")
                disconnected_connections.append(websocket)

        # Clean up disconnected connections
        for websocket in disconnected_connections:
            self.disconnect(websocket)

    async def broadcast_progress(self, progress: ScrapingProgress) -> None:
        """Broadcast scraping progress update.

        Args:
            progress: Scraping progress data
        """
        message = {
            "type": "progress",
            "job_id": progress.job_id,
            "progress": progress.progress,
            "message": progress.message,
            "matches_scraped": progress.matches_scraped,
            "timestamp": progress.timestamp.isoformat(),
        }

        await self.broadcast_to_job(progress.job_id, message)

    async def broadcast_job_status(
        self, job_id: str, status: str, message: str = ""
    ) -> None:
        """Broadcast job status update.

        Args:
            job_id: Job identifier
            status: New job status
            message: Optional status message
        """
        update_message = {
            "type": "status",
            "job_id": job_id,
            "status": status,
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        await self.broadcast_to_job(job_id, update_message)

    async def handle_client_message(self, websocket: WebSocket, message: str) -> None:
        """Handle incoming message from client.

        Args:
            websocket: Client WebSocket connection
            message: Received message
        """
        try:
            data: dict[str, Any] = json.loads(message)
            message_type = data.get("type")

            if message_type == "subscribe":
                job_id = data.get("job_id")
                if job_id:
                    await self._handle_subscription(websocket, job_id)
            elif message_type == "unsubscribe":
                job_id = data.get("job_id")
                if job_id:
                    await self._handle_unsubscription(websocket, job_id)
            elif message_type == "ping":
                await self.send_personal_message(
                    {
                        "type": "pong",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                    websocket,
                )
            else:
                logger.warning(f"Unknown message type: {message_type}")

        except json.JSONDecodeError:
            logger.error(f"Invalid JSON message: {message}")
            await self.send_personal_message(
                {"type": "error", "message": "Invalid JSON"}, websocket
            )
        except Exception as e:
            logger.error(f"Error handling client message: {e}")
            await self.send_personal_message(
                {"type": "error", "message": "Internal error"}, websocket
            )

    async def _handle_subscription(self, websocket: WebSocket, job_id: str) -> None:
        """Handle job subscription request.

        Args:
            websocket: Client WebSocket connection
            job_id: Job ID to subscribe to
        """
        metadata: dict[str, Any] = self.connection_metadata.get(websocket, {})
        current_job_id = metadata.get("job_id")

        # Unsubscribe from current job if different
        if current_job_id and current_job_id != job_id:
            await self._handle_unsubscription(websocket, current_job_id)

        # Subscribe to new job
        if job_id not in self.active_connections:
            self.active_connections[job_id] = []

        if websocket not in self.active_connections[job_id]:
            self.active_connections[job_id].append(websocket)
            metadata["job_id"] = job_id

            await self.send_personal_message(
                {
                    "type": "subscription_confirmed",
                    "job_id": job_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                websocket,
            )

            logger.info(
                f"Client {metadata.get('client_id')} subscribed to job {job_id}"
            )

    async def _handle_unsubscription(self, websocket: WebSocket, job_id: str) -> None:
        """Handle job unsubscription request.

        Args:
            websocket: Client WebSocket connection
            job_id: Job ID to unsubscribe from
        """
        if (
            job_id in self.active_connections
            and websocket in self.active_connections[job_id]
        ):
            self.active_connections[job_id].remove(websocket)

            if not self.active_connections[job_id]:
                del self.active_connections[job_id]

            metadata: dict[str, Any] = self.connection_metadata.get(websocket, {})
            if metadata.get("job_id") == job_id:
                metadata["job_id"] = None

            await self.send_personal_message(
                {
                    "type": "unsubscription_confirmed",
                    "job_id": job_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                websocket,
            )

            logger.info(
                f"Client {metadata.get('client_id')} unsubscribed from job {job_id}"
            )

    def get_connection_stats(self) -> dict[str, int]:
        """Get connection statistics.

        Returns:
            Dictionary with connection statistics
        """
        total_connections = len(self.connection_metadata)
        job_subscriptions = sum(
            len(connections) for connections in self.active_connections.values()
        )
        unique_jobs = len(self.active_connections)

        return {
            "total_connections": total_connections,
            "job_subscriptions": job_subscriptions,
            "unique_jobs": unique_jobs,
            "active_connections": len(
                [
                    conn
                    for conn in self.connection_metadata
                    if not conn.client_state.DISCONNECTED
                ]
            ),
        }


# Global connection manager instance
manager = ConnectionManager()


async def websocket_endpoint(
    websocket: WebSocket, client_id: str, job_id: str | None = None
) -> None:
    """WebSocket endpoint for progress updates.

    Args:
        websocket: WebSocket connection
        client_id: Unique client identifier
        job_id: Optional job ID to subscribe to immediately
    """
    await manager.connect(websocket, client_id, job_id)

    try:
        while True:
            # Receive and handle client messages
            message = await websocket.receive_text()
            await manager.handle_client_message(websocket, message)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {e}")
        manager.disconnect(websocket)
