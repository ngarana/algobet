/**
 * WebSocket hook for real-time fetch progress updates
 */

import { useEffect, useState, useRef, useCallback } from "react";

const WS_BASE_URL = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000";

export interface FetchProgress {
  type: "progress" | "status" | "connection" | "subscription_confirmed";
  job_id: string;
  progress?: number;
  status?: "pending" | "running" | "completed" | "failed" | "cancelled";
  matches_fetched?: number;
  matches_saved?: number;
  message?: string;
  current_page?: number;
  total_pages?: number;
  started_at?: string | null;
  completed_at?: string | null;
  timestamp?: string;
  error?: string;
}

export interface UseFetchProgressOptions {
  jobId?: string;
  enabled?: boolean;
  onProgress?: (progress: FetchProgress) => void;
  onError?: (error: Event) => void;
  onConnected?: () => void;
  onDisconnected?: () => void;
}

/**
 * Transform backend progress message to frontend FetchProgress type
 */
function transformProgress(data: Record<string, unknown>): FetchProgress {
  const rawStatus = data.status;
  const normalizedStatus =
    rawStatus === "pending" ||
    rawStatus === "running" ||
    rawStatus === "completed" ||
    rawStatus === "failed" ||
    rawStatus === "cancelled"
      ? rawStatus
      : undefined;

  return {
    type: data.type as FetchProgress["type"],
    job_id: data.job_id as string,
    progress: data.progress as number | undefined,
    status: normalizedStatus,
    matches_fetched: (data.matches_scraped || data.matches_fetched) as
      | number
      | undefined,
    matches_saved: data.matches_saved as number | undefined,
    message: data.message as string | undefined,
    current_page: data.current_page as number | undefined,
    total_pages: data.total_pages as number | undefined,
    started_at: data.started_at as string | null | undefined,
    completed_at: data.completed_at as string | null | undefined,
    timestamp: data.timestamp as string | undefined,
    error: data.error as string | undefined,
  };
}

export function useFetchProgress(options: UseFetchProgressOptions = {}) {
  const {
    jobId,
    onProgress,
    onError,
    onConnected,
    onDisconnected,
    enabled = true,
  } = options;

  const [isConnected, setIsConnected] = useState(false);
  const [currentProgress, setCurrentProgress] = useState<FetchProgress | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const onProgressRef = useRef(onProgress);
  const onErrorRef = useRef(onError);
  const onConnectedRef = useRef(onConnected);
  const onDisconnectedRef = useRef(onDisconnected);
  const reconnectAttemptsRef = useRef(0);
  const shouldReconnectRef = useRef(true);
  const maxReconnectAttempts = 5;
  const reconnectDelay = 3000;

  useEffect(() => {
    onProgressRef.current = onProgress;
    onErrorRef.current = onError;
    onConnectedRef.current = onConnected;
    onDisconnectedRef.current = onDisconnected;
  }, [onProgress, onError, onConnected, onDisconnected]);

  const connect = useCallback(() => {
    if (!jobId || !enabled) return;

    shouldReconnectRef.current = true;

    const ws = new WebSocket(`${WS_BASE_URL}/ws/scraping/${jobId}`);
    wsRef.current = ws;

    ws.onopen = () => {
      setIsConnected(true);
      reconnectAttemptsRef.current = 0;
      onConnectedRef.current?.();
    };

    ws.onmessage = (event) => {
      try {
        const rawData = JSON.parse(event.data);
        const progress = transformProgress(rawData);
        setCurrentProgress(progress);
        onProgressRef.current?.(progress);
      } catch (error) {
        console.error("Error parsing WebSocket message:", error);
      }
    };

    ws.onerror = (error) => {
      console.error("WebSocket error:", error);
      onErrorRef.current?.(error);
    };

    ws.onclose = () => {
      setIsConnected(false);
      onDisconnectedRef.current?.();

      // Attempt to reconnect if not intentionally closed
      if (
        shouldReconnectRef.current &&
        reconnectAttemptsRef.current < maxReconnectAttempts
      ) {
        reconnectAttemptsRef.current += 1;
        setTimeout(connect, reconnectDelay);
      }
    };
  }, [jobId, enabled]);

  const disconnect = useCallback(() => {
    shouldReconnectRef.current = false;
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
  }, []);

  const subscribe = useCallback((targetJobId: string) => {
    if (wsRef.current) {
      wsRef.current.send(
        JSON.stringify({
          type: "subscribe",
          job_id: targetJobId,
        })
      );
    }
  }, []);

  const unsubscribe = useCallback((targetJobId: string) => {
    if (wsRef.current) {
      wsRef.current.send(
        JSON.stringify({
          type: "unsubscribe",
          job_id: targetJobId,
        })
      );
    }
  }, []);

  useEffect(() => {
    connect();

    return () => {
      disconnect();
    };
  }, [connect, disconnect]);

  return {
    isConnected,
    currentProgress,
    connect,
    disconnect,
    subscribe,
    unsubscribe,
  };
}
