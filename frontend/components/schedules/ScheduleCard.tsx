'use client'

import { useState } from 'react'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import {
  PlayIcon,
  PauseIcon,
  TrashIcon,
  ClockIcon,
  CheckCircleIcon,
  XCircleIcon,
  MoreVerticalIcon,
  RefreshCwIcon,
} from 'lucide-react'
import type { ScheduledTask, TaskExecution } from '@/lib/api/schedules'

interface ScheduleCardProps {
  schedule: ScheduledTask
  lastExecution?: TaskExecution | null
  onRun?: () => void
  onToggleActive?: () => void
  onDelete?: () => void
  isLoading?: boolean
}

function getStatusBadge(isActive: boolean, lastExecution?: TaskExecution | null) {
  if (!isActive) {
    return (
      <Badge variant="outline">
        <PauseIcon className="w-3 h-3 mr-1" /> Inactive
      </Badge>
    )
  }

  if (!lastExecution) {
    return (
      <Badge variant="secondary">
        <ClockIcon className="w-3 h-3 mr-1" /> Pending
      </Badge>
    )
  }

  switch (lastExecution.status) {
    case 'completed':
      return (
        <Badge variant="success">
          <CheckCircleIcon className="w-3 h-3 mr-1" /> Active
        </Badge>
      )
    case 'failed':
      return (
        <Badge variant="destructive">
          <XCircleIcon className="w-3 h-3 mr-1" /> Error
        </Badge>
      )
    case 'running':
      return (
        <Badge variant="default">
          <RefreshCwIcon className="w-3 h-3 mr-1 animate-spin" /> Running
        </Badge>
      )
    default:
      return (
        <Badge variant="secondary">
          <ClockIcon className="w-3 h-3 mr-1" /> {lastExecution.status}
        </Badge>
      )
  }
}

function formatCronExpression(cron: string): string {
  const parts = cron.split(' ')
  if (parts.length !== 5) return cron

  const [minute, hour, dayOfMonth, month, dayOfWeek] = parts

  // Common patterns
  if (minute === '0' && hour === '6' && dayOfMonth === '*' && month === '*' && dayOfWeek === '*') {
    return 'Daily at 6:00 AM'
  }
  if (minute === '0' && hour === '18' && dayOfMonth === '*' && month === '*' && dayOfWeek === '*') {
    return 'Daily at 6:00 PM'
  }
  if (minute === '0' && hour === '7' && dayOfMonth === '*' && month === '*' && dayOfWeek === '*') {
    return 'Daily at 7:00 AM'
  }
  if (minute === '0' && hour === '3' && dayOfMonth === '*' && month === '*' && dayOfWeek === '1') {
    return 'Weekly on Monday at 3:00 AM'
  }

  return cron
}

function getTaskTypeLabel(taskType: string): string {
  switch (taskType) {
    case 'scrape_upcoming':
      return 'Scrape Upcoming'
    case 'scrape_results':
      return 'Scrape Results'
    case 'generate_predictions':
      return 'Generate Predictions'
    default:
      return taskType.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())
  }
}

export function ScheduleCard({
  schedule,
  lastExecution,
  onRun,
  onToggleActive,
  onDelete,
  isLoading = false,
}: ScheduleCardProps) {
  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-lg">{schedule.name}</CardTitle>
            <CardDescription>{getTaskTypeLabel(schedule.task_type)}</CardDescription>
          </div>
          {getStatusBadge(schedule.is_active, lastExecution)}
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-3">
          <div className="flex items-center gap-2 text-sm">
            <ClockIcon className="w-4 h-4 text-muted-foreground" />
            <span>{formatCronExpression(schedule.cron_expression)}</span>
            <code className="text-xs bg-muted px-1 rounded">{schedule.cron_expression}</code>
          </div>

          {schedule.description && (
            <p className="text-sm text-muted-foreground">{schedule.description}</p>
          )}

          {lastExecution && (
            <div className="text-xs text-muted-foreground space-y-1 pt-2 border-t">
              <p>Last run: {new Date(lastExecution.started_at).toLocaleString()}</p>
              {lastExecution.duration !== null && lastExecution.duration !== undefined && (
                <p>Duration: {lastExecution.duration.toFixed(2)}s</p>
              )}
              {lastExecution.error_message && (
                <p className="text-destructive">Error: {lastExecution.error_message}</p>
              )}
            </div>
          )}

          <div className="flex gap-2 pt-2">
            {onRun && (
              <Button
                size="sm"
                variant="default"
                onClick={onRun}
                disabled={isLoading || !schedule.is_active}
              >
                <PlayIcon className="w-4 h-4 mr-1" />
                Run Now
              </Button>
            )}
            {onToggleActive && (
              <Button
                size="sm"
                variant="outline"
                onClick={onToggleActive}
                disabled={isLoading}
              >
                {schedule.is_active ? (
                  <>
                    <PauseIcon className="w-4 h-4 mr-1" />
                    Disable
                  </>
                ) : (
                  <>
                    <PlayIcon className="w-4 h-4 mr-1" />
                    Enable
                  </>
                )}
              </Button>
            )}
            {onDelete && (
              <Button
                size="sm"
                variant="ghost"
                onClick={onDelete}
                disabled={isLoading}
                className="text-destructive hover:text-destructive"
              >
                <TrashIcon className="w-4 h-4" />
              </Button>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
