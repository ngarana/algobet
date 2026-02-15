/**
 * API client functions for schedule management
 */

import { apiGet, apiPost, apiPatch, apiDelete, buildQueryString } from './client'
import { z } from 'zod'

// Zod schemas for runtime validation
export const ScheduledTaskSchema = z.object({
  id: z.number(),
  name: z.string(),
  task_type: z.string(),
  cron_expression: z.string(),
  is_active: z.boolean(),
  parameters: z.record(z.unknown()),
  description: z.string().nullable(),
  created_at: z.string(),
  updated_at: z.string(),
})

export const TaskExecutionSchema = z.object({
  id: z.number(),
  task_id: z.number(),
  status: z.enum(['pending', 'running', 'completed', 'failed']),
  started_at: z.string(),
  completed_at: z.string().nullable(),
  duration: z.number().nullable(),
  result: z.record(z.unknown()).nullable(),
  error_message: z.string().nullable(),
})

export const ScheduleListResponseSchema = z.object({
  schedules: z.array(ScheduledTaskSchema),
  total: z.number(),
})

export const ExecutionHistoryResponseSchema = z.object({
  executions: z.array(TaskExecutionSchema),
  total: z.number(),
})

export const ScheduleCreateRequestSchema = z.object({
  name: z.string(),
  task_type: z.string(),
  cron_expression: z.string(),
  parameters: z.record(z.unknown()).optional(),
  description: z.string().optional(),
  is_active: z.boolean().optional(),
})

export const ScheduleUpdateRequestSchema = z.object({
  cron_expression: z.string().optional(),
  parameters: z.record(z.unknown()).optional(),
  description: z.string().optional(),
  is_active: z.boolean().optional(),
})

// Types derived from schemas
export type ScheduledTask = z.infer<typeof ScheduledTaskSchema>
export type TaskExecution = z.infer<typeof TaskExecutionSchema>
export type ScheduleListResponse = z.infer<typeof ScheduleListResponseSchema>
export type ExecutionHistoryResponse = z.infer<typeof ExecutionHistoryResponseSchema>
export type ScheduleCreateRequest = z.infer<typeof ScheduleCreateRequestSchema>
export type ScheduleUpdateRequest = z.infer<typeof ScheduleUpdateRequestSchema>

/**
 * Get all scheduled tasks
 */
export async function getSchedules(options?: {
  task_type?: string
  is_active?: boolean
}): Promise<ScheduleListResponse> {
  const params: Record<string, unknown> = {}
  if (options?.task_type) params.task_type = options.task_type
  if (options?.is_active !== undefined) params.is_active = String(options.is_active)
  
  const queryString = buildQueryString(params)
  return apiGet(`/schedules${queryString}`, ScheduleListResponseSchema)
}

/**
 * Get a specific scheduled task by ID
 */
export async function getSchedule(scheduleId: number): Promise<ScheduledTask> {
  return apiGet(`/schedules/${scheduleId}`, ScheduledTaskSchema)
}

/**
 * Get available task types
 */
export async function getTaskTypes(): Promise<string[]> {
  const schema = z.object({ task_types: z.array(z.string()) })
  const result = await apiGet('/schedules/task-types', schema)
  return result.task_types
}

/**
 * Create a new scheduled task
 */
export async function createSchedule(request: ScheduleCreateRequest): Promise<ScheduledTask> {
  return apiPost('/schedules', request, ScheduledTaskSchema)
}

/**
 * Update a scheduled task
 */
export async function updateSchedule(
  scheduleId: number,
  request: ScheduleUpdateRequest
): Promise<ScheduledTask> {
  return apiPatch(`/schedules/${scheduleId}`, request, ScheduledTaskSchema)
}

/**
 * Delete a scheduled task
 */
export async function deleteSchedule(scheduleId: number): Promise<void> {
  return apiDelete(`/schedules/${scheduleId}`)
}

/**
 * Run a scheduled task immediately
 */
export async function runScheduleNow(scheduleId: number): Promise<TaskExecution> {
  return apiPost(`/schedules/${scheduleId}/run`, {}, TaskExecutionSchema)
}

/**
 * Get execution history for a scheduled task
 */
export async function getExecutionHistory(
  scheduleId: number,
  limit: number = 100
): Promise<ExecutionHistoryResponse> {
  return apiGet(`/schedules/${scheduleId}/history?limit=${limit}`, ExecutionHistoryResponseSchema)
}

/**
 * Get the last execution for a scheduled task
 */
export async function getLastExecution(scheduleId: number): Promise<TaskExecution> {
  return apiGet(`/schedules/${scheduleId}/last-execution`, TaskExecutionSchema)
}