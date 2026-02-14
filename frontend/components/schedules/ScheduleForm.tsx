'use client'

import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { PlusIcon, Loader2Icon } from 'lucide-react'

interface ScheduleFormProps {
  taskTypes: string[]
  onSubmit: (data: {
    name: string
    task_type: string
    cron_expression: string
    parameters?: Record<string, unknown>
    description?: string
  }) => Promise<void>
  isLoading?: boolean
}

const COMMON_CRON_EXPRESSIONS = [
  { label: 'Daily at 6:00 AM', value: '0 6 * * *' },
  { label: 'Daily at 7:00 AM', value: '0 7 * * *' },
  { label: 'Daily at 6:00 PM', value: '0 18 * * *' },
  { label: 'Weekly on Monday at 3:00 AM', value: '0 3 * * 1' },
  { label: 'Every hour', value: '0 * * * *' },
  { label: 'Every 6 hours', value: '0 */6 * * *' },
  { label: 'Custom', value: 'custom' },
]

export function ScheduleForm({ taskTypes, onSubmit, isLoading = false }: ScheduleFormProps) {
  const [name, setName] = useState('')
  const [taskType, setTaskType] = useState('')
  const [cronPreset, setCronPreset] = useState('0 6 * * *')
  const [customCron, setCustomCron] = useState('')
  const [description, setDescription] = useState('')
  const [isSubmitting, setIsSubmitting] = useState(false)

  const cronExpression = cronPreset === 'custom' ? customCron : cronPreset

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!name || !taskType || !cronExpression) return

    setIsSubmitting(true)
    try {
      await onSubmit({
        name,
        task_type: taskType,
        cron_expression: cronExpression,
        description: description || undefined,
      })
      // Reset form
      setName('')
      setTaskType('')
      setCronPreset('0 6 * * *')
      setCustomCron('')
      setDescription('')
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <PlusIcon className="w-5 h-5" />
          Create New Schedule
        </CardTitle>
        <CardDescription>
          Set up a new automated task with cron scheduling
        </CardDescription>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="schedule-name">Schedule Name</Label>
            <Input
              id="schedule-name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g., daily_upcoming_scrape"
              required
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="task-type">Task Type</Label>
            <Select value={taskType} onValueChange={setTaskType} required>
              <SelectTrigger>
                <SelectValue placeholder="Select a task type" />
              </SelectTrigger>
              <SelectContent>
                {taskTypes.map((type) => (
                  <SelectItem key={type} value={type}>
                    {type.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label htmlFor="cron-preset">Schedule</Label>
            <Select value={cronPreset} onValueChange={setCronPreset}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {COMMON_CRON_EXPRESSIONS.map((preset) => (
                  <SelectItem key={preset.value} value={preset.value}>
                    {preset.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {cronPreset === 'custom' && (
            <div className="space-y-2">
              <Label htmlFor="custom-cron">Custom Cron Expression</Label>
              <Input
                id="custom-cron"
                value={customCron}
                onChange={(e) => setCustomCron(e.target.value)}
                placeholder="* * * * * (minute hour day month weekday)"
                required
              />
              <p className="text-xs text-muted-foreground">
                Format: minute hour day-of-month month day-of-week
              </p>
            </div>
          )}

          <div className="space-y-2">
            <Label htmlFor="description">Description (optional)</Label>
            <Input
              id="description"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Brief description of this schedule"
            />
          </div>

          <Button type="submit" disabled={isLoading || isSubmitting} className="w-full">
            {isSubmitting ? (
              <>
                <Loader2Icon className="mr-2 h-4 w-4 animate-spin" />
                Creating...
              </>
            ) : (
              <>
                <PlusIcon className="mr-2 h-4 w-4" />
                Create Schedule
              </>
            )}
          </Button>
        </form>
      </CardContent>
    </Card>
  )
}
