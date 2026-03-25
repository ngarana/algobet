import { describe, it, expect } from "vitest";
import { cn } from "../utils";

describe("cn utility function", () => {
  it("should merge tailwind classes correctly", () => {
    expect(cn("text-red-500", "text-blue-500")).toBe("text-blue-500");
  });

  it("should handle multiple class names", () => {
    expect(cn("flex", "items-center", "justify-center")).toBe(
      "flex items-center justify-center"
    );
  });

  it("should handle conditional classes with clsx", () => {
    const isActive = true;
    expect(cn("btn", isActive && "btn-active")).toBe("btn btn-active");
  });

  it("should handle false and null values", () => {
    expect(cn("btn", false, null, undefined)).toBe("btn");
  });

  it("should handle object notation", () => {
    expect(cn("btn", { "btn-active": true, "btn-disabled": false })).toBe(
      "btn btn-active"
    );
  });

  it("should prioritize tailwind classes in complex scenarios", () => {
    expect(cn("p-4 m-2", "p-8")).toBe("m-2 p-8");
  });

  it("should handle empty input", () => {
    expect(cn()).toBe("");
  });

  it("should handle array input", () => {
    expect(cn(["flex", "row"])).toBe("flex row");
  });
});
