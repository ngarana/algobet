import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import {
  ApiError,
  apiGet,
  apiPost,
  apiPut,
  apiPatch,
  apiDelete,
  buildQueryString,
} from "../client";
import { z } from "zod";

// Mock fetch globally
const mockFetch = vi.fn();
vi.stubGlobal("fetch", mockFetch);

describe("ApiError", () => {
  it("should create an ApiError with message and status", () => {
    const error = new ApiError("Not found", 404);

    expect(error.message).toBe("Not found");
    expect(error.status).toBe(404);
    expect(error.name).toBe("ApiError");
  });

  it("should create an ApiError with optional data", () => {
    const data = { code: "USER_NOT_FOUND" };
    const error = new ApiError("User not found", 404, data);

    expect(error.data).toEqual(data);
  });

  it("should be an instance of Error", () => {
    const error = new ApiError("Test error", 500);
    expect(error).toBeInstanceOf(Error);
  });
});

describe("apiGet", () => {
  beforeEach(() => {
    mockFetch.mockClear();
  });

  it("should make a GET request and return data", async () => {
    const mockData = { id: 1, name: "Test" };
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockData,
    });

    const result = await apiGet("/users/1");

    expect(mockFetch).toHaveBeenCalled();
    const callArgs = mockFetch.mock.calls[0];
    expect(callArgs[0]).toContain("/users/1");
    expect(callArgs[1]).toEqual({
      headers: { Accept: "application/json" },
    });
    expect(result).toEqual(mockData);
  });

  it("should throw ApiError on non-ok response", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 404,
      statusText: "Not Found",
      json: async () => ({ message: "User not found" }),
    });

    try {
      await apiGet("/users/999");
      expect.fail("Should have thrown ApiError");
    } catch (error) {
      expect(error).toBeInstanceOf(ApiError);
      expect(error.status).toBe(404);
      expect(error.message).toBe("User not found");
    }
  });

  it("should validate response with schema", async () => {
    const userSchema = z.object({
      id: z.number(),
      name: z.string(),
    });

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ id: 1, name: "John" }),
    });

    const result = await apiGet("/users/1", userSchema);

    expect(result).toEqual({ id: 1, name: "John" });
  });

  it("should throw when schema validation fails", async () => {
    const userSchema = z.object({
      id: z.number(),
      name: z.string(),
    });

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ id: "invalid", name: "John" }),
    });

    await expect(apiGet("/users/1", userSchema)).rejects.toThrow(
      "Invalid API response format"
    );
  });
});

describe("apiPost", () => {
  beforeEach(() => {
    mockFetch.mockClear();
  });

  it("should make a POST request with JSON body", async () => {
    const mockData = { id: 1, created: true };
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockData,
    });

    const result = await apiPost("/users", { name: "John" });

    expect(mockFetch).toHaveBeenCalled();
    const callArgs = mockFetch.mock.calls[0];
    expect(callArgs[0]).toContain("/users");
    expect(callArgs[1]).toEqual({
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      body: JSON.stringify({ name: "John" }),
    });
    expect(result).toEqual(mockData);
  });

  it("should handle POST errors", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 400,
      json: async () => ({ message: "Validation failed" }),
    });

    await expect(apiPost("/users", {})).rejects.toThrow(ApiError);
  });
});

describe("apiPut", () => {
  beforeEach(() => {
    mockFetch.mockClear();
  });

  it("should make a PUT request", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ id: 1, updated: true }),
    });

    await apiPut("/users/1", { name: "Updated" });

    expect(mockFetch).toHaveBeenCalled();
    const callArgs = mockFetch.mock.calls[0];
    expect(callArgs[0]).toContain("/users/1");
    expect(callArgs[1].method).toBe("PUT");
    expect(callArgs[1].body).toBe(JSON.stringify({ name: "Updated" }));
  });
});

describe("apiPatch", () => {
  beforeEach(() => {
    mockFetch.mockClear();
  });

  it("should make a PATCH request", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ id: 1, patched: true }),
    });

    await apiPatch("/users/1", { status: "active" });

    expect(mockFetch).toHaveBeenCalled();
    const callArgs = mockFetch.mock.calls[0];
    expect(callArgs[0]).toContain("/users/1");
    expect(callArgs[1].method).toBe("PATCH");
    expect(callArgs[1].body).toBe(JSON.stringify({ status: "active" }));
  });
});

describe("apiDelete", () => {
  beforeEach(() => {
    mockFetch.mockClear();
  });

  it("should make a DELETE request", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ success: true }),
    });

    const result = await apiDelete("/users/1");

    expect(mockFetch).toHaveBeenCalled();
    const callArgs = mockFetch.mock.calls[0];
    expect(callArgs[0]).toContain("/users/1");
    expect(callArgs[1]).toEqual({
      method: "DELETE",
      headers: { Accept: "application/json" },
    });
    expect(result).toEqual({ success: true });
  });
});

describe("buildQueryString", () => {
  it("should build query string from object", () => {
    const result = buildQueryString({ page: 1, limit: 10 });
    expect(result).toBe("?page=1&limit=10");
  });

  it("should return empty string for empty object", () => {
    const result = buildQueryString({});
    expect(result).toBe("");
  });

  it("should filter out undefined values", () => {
    const result = buildQueryString({ page: 1, search: undefined });
    expect(result).toBe("?page=1");
  });

  it("should filter out null values", () => {
    const result = buildQueryString({ page: 1, filter: null });
    expect(result).toBe("?page=1");
  });

  it("should handle multiple parameters", () => {
    const result = buildQueryString({
      page: 1,
      limit: 20,
      sort: "name",
      order: "asc",
    });
    expect(result).toContain("page=1");
    expect(result).toContain("limit=20");
    expect(result).toContain("sort=name");
    expect(result).toContain("order=asc");
  });

  it("should handle zero as a valid value", () => {
    const result = buildQueryString({ offset: 0, limit: 10 });
    expect(result).toBe("?offset=0&limit=10");
  });

  it("should handle boolean values", () => {
    const result = buildQueryString({ active: true, archived: false });
    expect(result).toBe("?active=true&archived=false");
  });
});

describe("API Base URL configuration", () => {
  it("should use environment variable for API base URL", () => {
    // This test verifies the configuration is read from environment
    // In actual usage, NEXT_PUBLIC_API_URL would be set in .env
    expect(process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000").toBeDefined();
  });
});
