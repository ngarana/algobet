/**
 * Base API client with error handling
 */

import { z } from 'zod'

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1'

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
    public data?: unknown
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

async function handleResponse<T>(response: Response, schema?: z.ZodType<T>): Promise<T> {
  if (!response.ok) {
    const errorData = await response.json().catch(() => null)
    throw new ApiError(
      errorData?.message || `HTTP ${response.status}: ${response.statusText}`,
      response.status,
      errorData
    )
  }

  const data = await response.json()
  
  if (schema) {
    const result = schema.safeParse(data)
    if (!result.success) {
      console.error('API response validation failed:', result.error)
      throw new ApiError('Invalid API response format', 500, result.error)
    }
    return result.data
  }

  return data as T
}

export async function apiGet<T>(
  endpoint: string,
  schema?: z.ZodType<T>
): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    headers: {
      'Accept': 'application/json',
    },
  })
  return handleResponse(response, schema)
}

export async function apiPost<T>(
  endpoint: string,
  body: unknown,
  schema?: z.ZodType<T>
): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
    },
    body: JSON.stringify(body),
  })
  return handleResponse(response, schema)
}

export async function apiPut<T>(
  endpoint: string,
  body: unknown,
  schema?: z.ZodType<T>
): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    method: 'PUT',
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
    },
    body: JSON.stringify(body),
  })
  return handleResponse(response, schema)
}

export async function apiPatch<T>(
  endpoint: string,
  body: unknown,
  schema?: z.ZodType<T>
): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    method: 'PATCH',
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
    },
    body: JSON.stringify(body),
  })
  return handleResponse(response, schema)
}

export async function apiDelete<T>(
  endpoint: string,
  schema?: z.ZodType<T>
): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    method: 'DELETE',
    headers: {
      'Accept': 'application/json',
    },
  })
  return handleResponse(response, schema)
}

// Helper function to build query strings
export function buildQueryString(params: object): string {
  const searchParams = new URLSearchParams()
  
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null) {
      searchParams.append(key, String(value))
    }
  })
  
  const queryString = searchParams.toString()
  return queryString ? `?${queryString}` : ''
}
