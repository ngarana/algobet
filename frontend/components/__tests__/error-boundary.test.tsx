import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { ErrorBoundary } from "../error-boundary";

describe("ErrorBoundary", () => {
  const mockError = {
    message: "Test error message",
    digest: "test-digest-123",
  };

  const mockReset = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
    console.error = vi.fn(); // Mock console.error to avoid noise in tests
  });

  it("should render error boundary with error message", () => {
    render(<ErrorBoundary error={mockError} reset={mockReset} />);

    expect(screen.getByText("Something went wrong")).toBeInTheDocument();
    expect(screen.getByText(/Test error message/)).toBeInTheDocument();
  });

  it("should display error digest when available", () => {
    render(<ErrorBoundary error={mockError} reset={mockReset} />);

    expect(screen.getByText(/Error ID:/)).toBeInTheDocument();
    expect(screen.getByText(/test-digest-123/)).toBeInTheDocument();
  });

  it("should not display error digest when not available", () => {
    const errorWithoutDigest = { message: "Test error" };
    render(<ErrorBoundary error={errorWithoutDigest as Error} reset={mockReset} />);

    expect(screen.queryByText(/Error ID:/)).not.toBeInTheDocument();
  });

  it("should render alert icon", () => {
    render(<ErrorBoundary error={mockError} reset={mockReset} />);

    // Check if the AlertTriangle icon is rendered (by checking the SVG)
    const svg = document.querySelector("svg");
    expect(svg).toBeInTheDocument();
  });

  it('should call reset function when "Try again" button is clicked', () => {
    render(<ErrorBoundary error={mockError} reset={mockReset} />);

    const tryAgainButton = screen.getByRole("button", { name: /try again/i });
    fireEvent.click(tryAgainButton);

    expect(mockReset).toHaveBeenCalledTimes(1);
  });

  it('should redirect to home when "Go home" button is clicked', () => {
    render(<ErrorBoundary error={mockError} reset={mockReset} />);

    const goHomeButton = screen.getByRole("button", { name: /go home/i });

    // Mock window.location.href
    const originalHref = window.location.href;
    Object.defineProperty(window, "location", {
      value: { href: "" },
      writable: true,
    });

    fireEvent.click(goHomeButton);

    expect(window.location.href).toBe("/");

    // Restore original location
    Object.defineProperty(window, "location", {
      value: { href: originalHref },
      writable: true,
    });
  });

  it("should have proper styling classes", () => {
    render(<ErrorBoundary error={mockError} reset={mockReset} />);

    // Check for main container classes
    const container = screen.getByText("Something went wrong").closest(".flex");
    expect(container).toBeInTheDocument();
  });

  it("should log error to console on mount", () => {
    render(<ErrorBoundary error={mockError} reset={mockReset} />);

    expect(console.error).toHaveBeenCalledWith(
      "Error caught by boundary:",
      expect.objectContaining({
        message: "Test error message",
        digest: "test-digest-123",
      })
    );
  });

  it("should use default message when error message is empty", () => {
    const emptyError = { message: "", digest: "test-123" };
    render(<ErrorBoundary error={emptyError as Error} reset={mockReset} />);

    // Should show fallback message
    expect(screen.getByText(/An unexpected error occurred/i)).toBeInTheDocument();
  });

  it("should render buttons with correct variants", () => {
    render(<ErrorBoundary error={mockError} reset={mockReset} />);

    const tryAgainButton = screen.getByRole("button", { name: /try again/i });
    const goHomeButton = screen.getByRole("button", { name: /go home/i });

    // Check that buttons exist and have different styles
    expect(tryAgainButton).toHaveClass("bg-primary");
    expect(goHomeButton).toHaveClass("border");
  });

  it("should be accessible with proper ARIA attributes", () => {
    render(<ErrorBoundary error={mockError} reset={mockReset} />);

    // Buttons should be accessible
    const tryAgainButton = screen.getByRole("button", { name: /try again/i });
    const goHomeButton = screen.getByRole("button", { name: /go home/i });

    expect(tryAgainButton).toBeInTheDocument();
    expect(goHomeButton).toBeInTheDocument();
  });
});
