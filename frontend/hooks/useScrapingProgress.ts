// WebSocket hook for real-time scraping progress updates

import { useEffect, useState, useRef, useCallback } from 'react';

const WS_BASE_URL = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000/api/v1';

export interface ScrapingProgress {
  type: 'progress' | 'status' | 'connection' | 'subscription_confirmed';
  job_id: string;
  progress?: number;
  status?: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  matches_scraped?: number;
  message?: string;
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
  const maxReconnectAttempts = 5;
  const reconnectDelay = 3000;

  const connect = useCallback(() => {
    if (!jobId || !enabled) return;

    const ws = new WebSocket(`${WS_BASE_URL}/ws/scraping/${jobId}`);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log(`WebSocket connected for job ${jobId}`);
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
        console.error('Error parsing WebSocket message:', error);
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      onError?.(error);
    };

    ws.onclose = () => {
      console.log(`WebSocket disconnected for job ${jobId}`);
      setIsConnected(false);
      onDisconnected?.();

      // Attempt to reconnect if not intentionally closed
      if (reconnectAttemptsRef.current < maxReconnectAttempts) {
        reconnectAttemptsRef.current += 1;
        console.log(`Reconnecting... Attempt ${reconnectAttemptsRef.current}`);
        setTimeout(connect, reconnectDelay);
      }
    };
  }, [jobId, enabled, onProgress, onError, onConnected, onDisconnected]);

  const disconnect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
  }, []);

  const subscribe = useCallback((targetJobId: string) => {
    if (wsRef.current) {
      wsRef.current.send(JSON.stringify({
        action: 'subscribe',
        job_id: targetJobId,
      }));
    }
  }, []);

  const unsubscribe = useCallback((targetJobId: string) => {
    if (wsRef.current) {
      wsRef.current.send(JSON.stringify({
        action: 'unsubscribe',
        job_id: targetJobId,
      }));
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
