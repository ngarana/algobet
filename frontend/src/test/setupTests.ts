import "@testing-library/jest-dom";
import { cleanup } from "@testing-library/react";
import { afterEach, vi } from "vitest";
import React from "react";

// Make React available globally for JSX
vi.stubGlobal("React", React);

// Cleanup after each test
afterEach(() => {
  cleanup();
});

// Mock Next.js router
vi.mock("next/navigation", () => ({
  useRouter() {
    return {
      push: vi.fn(),
      replace: vi.fn(),
      prefetch: vi.fn(),
      back: vi.fn(),
      forward: vi.fn(),
      refresh: vi.fn(),
    };
  },
  usePathname() {
    return "/";
  },
  useSearchParams() {
    return new URLSearchParams();
  },
}));

// Mock environment variables
process.env.NEXT_PUBLIC_API_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
process.env.NEXT_PUBLIC_WS_URL =
  process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000";
