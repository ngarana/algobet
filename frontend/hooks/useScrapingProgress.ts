// WebSocket hook for real-time scraping progress updates

import { useEffect, useState, useRef, useCallback } from "react";

const WS_BASE_URL = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000";

export interface ScrapingProgress {
  type: "progress" | "status" | "connection" | "subscription_confirmed";
  job_id: string;
  progress?: number;
  status?: "pending" | "running" | "completed" | "failed" | "cancelled";
  matches_scraped?: number;
  matches_saved?: number;
  message?: string;
  current_page?: number;
  total_pages?: number;
  started_at?: string | null;
  completed_at?: string | null;
  timestamp?: string;
  error?: string;
}

export interface UseScrapingProgressOptions {
  jobId?: string;
  onProgress?: (progress: ScrapingProgress) => void;
  onError?: (error: Event) => void;
  onConnected?: () => void;
  onDisconnected?: () => void;
  enabled?: boolean;
}

export function useScrapingProgress(options: UseScrapingProgressOptions = {}) {
  const {
    jobId,
    onProgress,
    onError,
    onConnected,
    onDisconnected,
    enabled = true,
  } = options;

  const [isConnected, setIsConnected] = useState(false);
  const [currentProgress, setCurrentProgress] = useState<ScrapingProgress | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectAttemptsRef = useRef(0);
  const shouldReconnectRef = useRef(true);
  const maxReconnectAttempts = 5;
  const reconnectDelay = 3000;

  const connect = useCallback(() => {
    if (!jobId || !enabled) return;

    shouldReconnectRef.current = true;

    const ws = new WebSocket(`${WS_BASE_URL}/ws/scraping/${jobId}`);
    wsRef.current = ws;

    ws.onopen = () => {
      setIsConnected(true);
      reconnectAttemptsRef.current = 0;
      onConnected?.();
    };

    ws.onmessage = (event) => {
      try {
        const progress: ScrapingProgress = JSON.parse(event.data);
        setCurrentProgress(progress);
        onProgress?.(progress);
      } catch (error) {
        console.error("Error parsing WebSocket message:", error);
      }
    };

    ws.onerror = (error) => {
      console.error("WebSocket error:", error);
      onError?.(error);
    };

    ws.onclose = () => {
      setIsConnected(false);
      onDisconnected?.();

      // Attempt to reconnect if not intentionally closed
      if (
        shouldReconnectRef.current &&
        reconnectAttemptsRef.current < maxReconnectAttempts
      ) {
        reconnectAttemptsRef.current += 1;
        setTimeout(connect, reconnectDelay);
      }
    };
  }, [jobId, enabled, onProgress, onError, onConnected, onDisconnected]);

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
