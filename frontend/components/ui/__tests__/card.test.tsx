import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import {
  Card,
  CardHeader,
  CardFooter,
  CardTitle,
  CardDescription,
  CardContent,
} from "../card";

describe("Card", () => {
  it("should render a card element", () => {
    render(<Card>Card content</Card>);
    const card = screen.getByText("Card content");
    expect(card).toBeInTheDocument();
    expect(card.tagName).toBe("DIV");
  });

  it("should apply base card styles", () => {
    const { container } = render(<Card>Content</Card>);
    expect(container.firstChild).toHaveClass("rounded-lg");
    expect(container.firstChild).toHaveClass("border");
    expect(container.firstChild).toHaveClass("bg-card");
    expect(container.firstChild).toHaveClass("shadow-sm");
  });

  it("should accept custom className", () => {
    const { container } = render(<Card className="custom-card">Content</Card>);
    expect(container.firstChild).toHaveClass("custom-card");
  });

  it("should forward ref correctly", () => {
    const ref = vi.fn();
    render(<Card ref={ref}>Content</Card>);
    expect(ref).toHaveBeenCalledWith(expect.any(HTMLDivElement));
  });
});

describe("CardHeader", () => {
  it("should render a card header", () => {
    render(<CardHeader>Header</CardHeader>);
    expect(screen.getByText("Header")).toBeInTheDocument();
  });

  it("should apply header styles", () => {
    const { container } = render(<CardHeader>Header</CardHeader>);
    expect(container.firstChild).toHaveClass("flex");
    expect(container.firstChild).toHaveClass("flex-col");
    expect(container.firstChild).toHaveClass("space-y-1.5");
    expect(container.firstChild).toHaveClass("p-6");
  });

  it("should accept custom className", () => {
    const { container } = render(
      <CardHeader className="custom-header">Header</CardHeader>
    );
    expect(container.firstChild).toHaveClass("custom-header");
  });

  it("should forward ref", () => {
    const ref = vi.fn();
    render(<CardHeader ref={ref}>Header</CardHeader>);
    expect(ref).toHaveBeenCalledWith(expect.any(HTMLDivElement));
  });
});

describe("CardTitle", () => {
  it("should render as h3 element", () => {
    render(<CardTitle>Title</CardTitle>);
    const title = screen.getByText("Title");
    expect(title.tagName).toBe("H3");
  });

  it("should apply title styles", () => {
    const { container } = render(<CardTitle>Title</CardTitle>);
    expect(container.firstChild).toHaveClass("text-2xl");
    expect(container.firstChild).toHaveClass("font-semibold");
    expect(container.firstChild).toHaveClass("leading-none");
  });

  it("should accept custom className", () => {
    const { container } = render(<CardTitle className="custom-title">Title</CardTitle>);
    expect(container.firstChild).toHaveClass("custom-title");
  });

  it("should forward ref", () => {
    const ref = vi.fn();
    render(<CardTitle ref={ref}>Title</CardTitle>);
    expect(ref).toHaveBeenCalledWith(expect.any(HTMLElement));
  });
});

describe("CardDescription", () => {
  it("should render description paragraph", () => {
    render(<CardDescription>Description text</CardDescription>);
    expect(screen.getByText("Description text")).toBeInTheDocument();
  });

  it("should apply description styles", () => {
    const { container } = render(<CardDescription>Description</CardDescription>);
    expect(container.firstChild).toHaveClass("text-sm");
    expect(container.firstChild).toHaveClass("text-muted-foreground");
  });

  it("should accept custom className", () => {
    const { container } = render(
      <CardDescription className="custom-desc">Description</CardDescription>
    );
    expect(container.firstChild).toHaveClass("custom-desc");
  });

  it("should forward ref", () => {
    const ref = vi.fn();
    render(<CardDescription ref={ref}>Description</CardDescription>);
    expect(ref).toHaveBeenCalledWith(expect.any(HTMLParagraphElement));
  });
});

describe("CardContent", () => {
  it("should render card content", () => {
    render(<CardContent>Content here</CardContent>);
    expect(screen.getByText("Content here")).toBeInTheDocument();
  });

  it("should apply content styles", () => {
    const { container } = render(<CardContent>Content</CardContent>);
    expect(container.firstChild).toHaveClass("p-6");
    expect(container.firstChild).toHaveClass("pt-0");
  });

  it("should accept custom className", () => {
    const { container } = render(
      <CardContent className="custom-content">Content</CardContent>
    );
    expect(container.firstChild).toHaveClass("custom-content");
  });

  it("should forward ref", () => {
    const ref = vi.fn();
    render(<CardContent ref={ref}>Content</CardContent>);
    expect(ref).toHaveBeenCalledWith(expect.any(HTMLDivElement));
  });
});

describe("CardFooter", () => {
  it("should render card footer", () => {
    render(<CardFooter>Footer</CardFooter>);
    expect(screen.getByText("Footer")).toBeInTheDocument();
  });

  it("should apply footer styles", () => {
    const { container } = render(<CardFooter>Footer</CardFooter>);
    expect(container.firstChild).toHaveClass("flex");
    expect(container.firstChild).toHaveClass("items-center");
    expect(container.firstChild).toHaveClass("p-6");
    expect(container.firstChild).toHaveClass("pt-0");
  });

  it("should accept custom className", () => {
    const { container } = render(
      <CardFooter className="custom-footer">Footer</CardFooter>
    );
    expect(container.firstChild).toHaveClass("custom-footer");
  });

  it("should forward ref", () => {
    const ref = vi.fn();
    render(<CardFooter ref={ref}>Footer</CardFooter>);
    expect(ref).toHaveBeenCalledWith(expect.any(HTMLDivElement));
  });
});

describe("Card composition", () => {
  it("should render complete card structure", () => {
    render(
      <Card>
        <CardHeader>
          <CardTitle>Card Title</CardTitle>
          <CardDescription>Card Description</CardDescription>
        </CardHeader>
        <CardContent>Card Content</CardContent>
        <CardFooter>Card Footer</CardFooter>
      </Card>
    );

    expect(screen.getByText("Card Title")).toBeInTheDocument();
    expect(screen.getByText("Card Description")).toBeInTheDocument();
    expect(screen.getByText("Card Content")).toBeInTheDocument();
    expect(screen.getByText("Card Footer")).toBeInTheDocument();
  });

  it("should maintain proper hierarchy", () => {
    const { container } = render(
      <Card data-testid="card">
        <CardHeader data-testid="header">
          <CardTitle data-testid="title">Title</CardTitle>
        </CardHeader>
        <CardContent data-testid="content">Content</CardContent>
      </Card>
    );

    const card = screen.getByTestId("card");
    const header = screen.getByTestId("header");
    const content = screen.getByTestId("content");

    expect(header.parentElement).toBe(card);
    expect(content.parentElement).toBe(card);
  });
});
