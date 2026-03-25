import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Button, buttonVariants } from "../button";

describe("Button", () => {
  describe("rendering", () => {
    it("should render a button element", () => {
      render(<Button>Click me</Button>);
      const button = screen.getByRole("button", { name: /click me/i });
      expect(button).toBeInTheDocument();
      expect(button.tagName).toBe("BUTTON");
    });

    it("should render children correctly", () => {
      render(<Button>Test Button</Button>);
      expect(screen.getByText("Test Button")).toBeInTheDocument();
    });

    it("should apply custom className", () => {
      const { container } = render(<Button className="custom-class">Click me</Button>);
      expect(container.firstChild).toHaveClass("custom-class");
    });

    it("should forward ref correctly", () => {
      const ref = vi.fn();
      render(<Button ref={ref}>Click me</Button>);
      expect(ref).toHaveBeenCalledWith(expect.any(HTMLButtonElement));
    });
  });

  describe("variants", () => {
    it("should apply default variant styles", () => {
      const { container } = render(<Button>Default</Button>);
      expect(container.firstChild).toHaveClass("bg-primary");
      expect(container.firstChild).toHaveClass("text-primary-foreground");
    });

    it("should apply destructive variant", () => {
      const { container } = render(<Button variant="destructive">Destructive</Button>);
      expect(container.firstChild).toHaveClass("bg-destructive");
    });

    it("should apply outline variant", () => {
      const { container } = render(<Button variant="outline">Outline</Button>);
      expect(container.firstChild).toHaveClass("border");
      expect(container.firstChild).toHaveClass("border-input");
    });

    it("should apply ghost variant", () => {
      const { container } = render(<Button variant="ghost">Ghost</Button>);
      expect(container.firstChild).toHaveClass("hover:bg-accent");
    });

    it("should apply link variant", () => {
      const { container } = render(<Button variant="link">Link</Button>);
      expect(container.firstChild).toHaveClass("underline-offset-4");
    });
  });

  describe("sizes", () => {
    it("should apply default size", () => {
      const { container } = render(<Button>Default Size</Button>);
      expect(container.firstChild).toHaveClass("h-10");
      expect(container.firstChild).toHaveClass("px-4");
    });

    it("should apply small size", () => {
      const { container } = render(<Button size="sm">Small</Button>);
      expect(container.firstChild).toHaveClass("h-9");
      expect(container.firstChild).toHaveClass("px-3");
    });

    it("should apply large size", () => {
      const { container } = render(<Button size="lg">Large</Button>);
      expect(container.firstChild).toHaveClass("h-11");
      expect(container.firstChild).toHaveClass("px-8");
    });

    it("should apply icon size", () => {
      const { container } = render(<Button size="icon">Icon</Button>);
      expect(container.firstChild).toHaveClass("h-10");
      expect(container.firstChild).toHaveClass("w-10");
    });
  });

  describe("disabled state", () => {
    it("should apply disabled styles", () => {
      const { container } = render(<Button disabled>Disabled</Button>);
      expect(container.firstChild).toHaveAttribute("disabled");
      expect(container.firstChild).toHaveClass("disabled:pointer-events-none");
      expect(container.firstChild).toHaveClass("disabled:opacity-50");
    });

    it("should not be clickable when disabled", async () => {
      const onClick = vi.fn();
      const user = userEvent.setup();

      render(
        <Button disabled onClick={onClick}>
          Disabled
        </Button>
      );
      const button = screen.getByRole("button");

      await user.click(button);
      expect(onClick).not.toHaveBeenCalled();
    });
  });

  describe("asChild prop", () => {
    it("should render as child component when asChild is true", () => {
      render(
        <Button asChild>
          <a href="/test">Link</a>
        </Button>
      );
      const link = screen.getByRole("link", { name: /link/i });
      expect(link).toBeInTheDocument();
      expect(link.tagName).toBe("A");
    });

    it("should apply button styles to child component", () => {
      render(
        <Button asChild className="custom-button">
          <a href="/test">Link</a>
        </Button>
      );
      const link = screen.getByRole("link");
      expect(link).toHaveClass("custom-button");
    });
  });

  describe("interactions", () => {
    it("should call onClick handler when clicked", async () => {
      const onClick = vi.fn();
      const user = userEvent.setup();

      render(<Button onClick={onClick}>Click me</Button>);
      const button = screen.getByRole("button", { name: /click me/i });

      await user.click(button);
      expect(onClick).toHaveBeenCalledTimes(1);
    });

    it("should call onClick handler multiple times when clicked multiple times", async () => {
      const onClick = vi.fn();
      const user = userEvent.setup();

      render(<Button onClick={onClick}>Click me</Button>);
      const button = screen.getByRole("button", { name: /click me/i });

      await user.click(button);
      await user.click(button);
      await user.click(button);

      expect(onClick).toHaveBeenCalledTimes(3);
    });
  });

  describe("accessibility", () => {
    it("should have proper focus styles", () => {
      const { container } = render(<Button>Focusable</Button>);
      expect(container.firstChild).toHaveClass("focus-visible:outline-none");
      expect(container.firstChild).toHaveClass("focus-visible:ring-2");
    });

    it("should support aria-label", () => {
      render(<Button aria-label="Close dialog">X</Button>);
      const button = screen.getByLabelText("Close dialog");
      expect(button).toBeInTheDocument();
    });

    it("should support aria-describedby", () => {
      render(
        <>
          <span id="description">Button description</span>
          <Button aria-describedby="description">Help</Button>
        </>
      );
      const button = screen.getByRole("button", { name: /help/i });
      expect(button).toHaveAttribute("aria-describedby", "description");
    });
  });
});

describe("buttonVariants", () => {
  it("should generate correct class names for variants", () => {
    expect(buttonVariants({ variant: "default" })).toContain("bg-primary");
    expect(buttonVariants({ variant: "destructive" })).toContain("bg-destructive");
  });

  it("should generate correct class names for sizes", () => {
    expect(buttonVariants({ size: "default" })).toContain("h-10");
    expect(buttonVariants({ size: "sm" })).toContain("h-9");
    expect(buttonVariants({ size: "lg" })).toContain("h-11");
  });

  it("should combine multiple classes", () => {
    const result = buttonVariants({
      variant: "primary",
      size: "default",
      className: "custom-class",
    });
    expect(result).toContain("custom-class");
  });
});
