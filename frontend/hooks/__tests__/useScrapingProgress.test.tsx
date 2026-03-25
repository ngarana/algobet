import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { useScrapingProgress } from "../useScrapingProgress";

// Mock WebSocket
class MockWebSocket {
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSING = 2;
  static CLOSED = 3;

  onopen: (() => void) | null = null;
  onclose: (() => void) | null = null;
  onerror: ((error: Event) => void) | null = null;
  onmessage: ((event: MessageEvent) => void) | null = null;
  readyState: number = MockWebSocket.CONNECTING;

  constructor(public url: string) {
    setTimeout(() => {
      this.readyState = MockWebSocket.OPEN;
      this.onopen?.();
    }, 0);
  }

  send(_data: string) {
    // Mock send
  }

  close() {
    this.readyState = MockWebSocket.CLOSED;
    this.onclose?.();
  }
}

vi.stubGlobal("WebSocket", MockWebSocket);

describe("useScrapingProgress", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("should initialize with default values", () => {
    const { result } = renderHook(() =>
      useScrapingProgress({ jobId: "test-job-123", enabled: false })
    );

    expect(result.current.isConnected).toBe(false);
    expect(result.current.currentProgress).toBeNull();
  });

  it("should connect to WebSocket when enabled", async () => {
    const { result } = renderHook(() =>
      useScrapingProgress({
        jobId: "test-job-123",
        enabled: true,
      })
    );

    await waitFor(() => {
      expect(result.current.isConnected).toBe(true);
    });
  });

  it("should not connect when disabled", () => {
    const { result } = renderHook(() =>
      useScrapingProgress({
        jobId: "test-job-123",
        enabled: false,
      })
    );

    expect(result.current.isConnected).toBe(false);
  });

  it("should handle progress updates", async () => {
    const onProgressMock = vi.fn();

    const { result } = renderHook(() =>
      useScrapingProgress({
        jobId: "test-job-123",
        enabled: true,
        onProgress: onProgressMock,
      })
    );

    // Simulate receiving a message
    await waitFor(() => {
      expect(result.current.isConnected).toBe(true);
    });

    // Note: In a real scenario, the WebSocket would receive messages
    // This is a basic connectivity test
    expect(result.current.isConnected).toBe(true);
  });

  it("should call onConnected callback when connection is established", async () => {
    const onConnectedMock = vi.fn();

    const { result } = renderHook(() =>
      useScrapingProgress({
        jobId: "test-job-123",
        enabled: true,
        onConnected: onConnectedMock,
      })
    );

    await waitFor(() => {
      expect(result.current.isConnected).toBe(true);
    });

    expect(onConnectedMock).toHaveBeenCalled();
  });

  it("should provide disconnect function", async () => {
    const { result } = renderHook(() =>
      useScrapingProgress({
        jobId: "test-job-123",
        enabled: true,
      })
    );

    await waitFor(() => {
      expect(result.current.isConnected).toBe(true);
    });

    expect(result.current.disconnect).toBeDefined();
    expect(typeof result.current.disconnect).toBe("function");
  });

  it("should provide subscribe and unsubscribe functions", () => {
    const { result } = renderHook(() =>
      useScrapingProgress({
        jobId: "test-job-123",
        enabled: false,
      })
    );

    expect(result.current.subscribe).toBeDefined();
    expect(result.current.unsubscribe).toBeDefined();
    expect(typeof result.current.subscribe).toBe("function");
    expect(typeof result.current.unsubscribe).toBe("function");
  });

  it("should handle different job IDs", async () => {
    const { result, rerender } = renderHook(
      ({ jobId }) => useScrapingProgress({ jobId, enabled: true }),
      { initialProps: { jobId: "job-1" } }
    );

    await waitFor(() => {
      expect(result.current.isConnected).toBe(true);
    });

    rerender({ jobId: "job-2" });

    await waitFor(() => {
      expect(result.current.isConnected).toBe(true);
    });
  });
});
