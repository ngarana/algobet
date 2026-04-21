/**
 * Base API client with error handling
 */

import { z } from "zod";

// Use relative URL to leverage Next.js API rewrites
// This works both in Docker (with rewrites) and outside Docker (direct connection)
const API_BASE_URL =
  typeof window === "undefined"
    ? // Server-side: use internal URL for Docker communication
      process.env.API_INTERNAL_URL || "http://localhost:8010"
    : // Client-side: use relative URL to leverage Next.js rewrites
      "/api/v1";

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
    public data?: unknown
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function handleResponse<T>(
  response: Response,
  schema?: z.ZodType<T>
): Promise<T> {
  if (!response.ok) {
    const errorData = await response.json().catch(() => null);
    throw new ApiError(
      errorData?.message || `HTTP ${response.status}: ${response.statusText}`,
      response.status,
      errorData
    );
  }

  // Handle empty responses
  const contentType = response.headers.get("content-type");
  if (!contentType || !contentType.includes("application/json")) {
    return {} as T;
  }

  const text = await response.text();
  if (!text || text.trim() === "") {
    return {} as T;
  }

  let data: unknown;
  try {
    data = JSON.parse(text);
  } catch {
    return {} as T;
  }

  // Handle null data
  if (data === null || data === undefined) {
    return {} as T;
  }

  if (schema) {
    const result = schema.safeParse(data);
    if (!result.success) {
      console.warn("API response validation warning:", result.error);
      // Return data anyway instead of throwing
      return data as T;
    }
    return result.data;
  }

  return data as T;
}

export async function apiGet<T>(endpoint: string, schema?: z.ZodType<T>): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    headers: {
      Accept: "application/json",
    },
  });
  return handleResponse(response, schema);
}

export async function apiPost<T>(
  endpoint: string,
  body: unknown,
  schema?: z.ZodType<T>
): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json",
    },
    body: JSON.stringify(body),
  });
  return handleResponse(response, schema);
}

export async function apiPut<T>(
  endpoint: string,
  body: unknown,
  schema?: z.ZodType<T>
): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    method: "PUT",
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json",
    },
    body: JSON.stringify(body),
  });
  return handleResponse(response, schema);
}

export async function apiPatch<T>(
  endpoint: string,
  body: unknown,
  schema?: z.ZodType<T>
): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    method: "PATCH",
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json",
    },
    body: JSON.stringify(body),
  });
  return handleResponse(response, schema);
}

export async function apiDelete<T>(
  endpoint: string,
  schema?: z.ZodType<T>
): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    method: "DELETE",
    headers: {
      Accept: "application/json",
    },
  });
  return handleResponse(response, schema);
}

// Helper function to build query strings
export function buildQueryString(params: object): string {
  const searchParams = new URLSearchParams();

  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null) {
      searchParams.append(key, String(value));
    }
  });

  const queryString = searchParams.toString();
  return queryString ? `?${queryString}` : "";
}
