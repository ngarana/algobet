'use client'

import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Progress, ProgressValue } from '@/components/ui/progress'
import {
  CheckCircleIcon,
  XCircleIcon,
  ClockIcon,
  Loader2Icon,
  PauseIcon,
} from 'lucide-react'

export interface ScrapingJob {
  id: string
  job_type: string
  url: string
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'
  created_at: string
  progress: ScrapingProgress | null
}

export interface ScrapingProgress {
  job_id: string
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'
  current_page: number
  total_pages: number
  matches_scraped: number
  matches_saved: number
  message: string
  error: string | null
  started_at: string | null
  completed_at: string | null
}

interface ScrapingJobCardProps {
  job: ScrapingJob
}

function getStatusBadge(status: string) {
  switch (status) {
    case 'pending':
      return (
        <Badge variant="secondary">
          <ClockIcon className="w-3 h-3 mr-1" /> Pending
        </Badge>
      )
    case 'running':
      return (
        <Badge variant="default">
          <Loader2Icon className="w-3 h-3 mr-1 animate-spin" /> Running
        </Badge>
      )
    case 'completed':
      return (
        <Badge variant="success">
          <CheckCircleIcon className="w-3 h-3 mr-1" /> Completed
        </Badge>
      )
    case 'failed':
      return (
        <Badge variant="destructive">
          <XCircleIcon className="w-3 h-3 mr-1" /> Failed
        </Badge>
      )
    case 'cancelled':
      return (
        <Badge variant="outline">
          <PauseIcon className="w-3 h-3 mr-1" /> Cancelled
        </Badge>
      )
    default:
      return <Badge variant="outline">{status}</Badge>
  }
}

export function ScrapingJobCard({ job }: ScrapingJobCardProps) {
  const progressPercent =
    job.progress && job.progress.total_pages > 0
      ? (job.progress.current_page / job.progress.total_pages) * 100
      : 0

  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg">{job.job_type === 'upcoming' ? 'Upcoming Matches' : 'Historical Results'}</CardTitle>
          {getStatusBadge(job.status)}
        </div>
        <CardDescription className="truncate max-w-md">{job.url}</CardDescription>
      </CardHeader>
      <CardContent>
        {job.progress ? (
          <div className="space-y-3">
            <div className="flex items-center justify-between text-sm">
              <span className="text-muted-foreground">Progress</span>
              <span className="font-medium">
                {job.progress.current_page} / {job.progress.total_pages || '?'} pages
              </span>
            </div>
            <Progress>
              <ProgressValue value={progressPercent} />
            </Progress>
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <p className="text-muted-foreground">Matches Scraped</p>
                <p className="font-semibold text-lg">{job.progress.matches_scraped}</p>
              </div>
              <div>
                <p className="text-muted-foreground">Matches Saved</p>
                <p className="font-semibold text-lg">{job.progress.matches_saved}</p>
              </div>
            </div>
            {job.progress.message && (
              <p className="text-sm text-muted-foreground border-t pt-2">
                {job.progress.message}
              </p>
            )}
            {job.progress.error && (
              <p className="text-sm text-destructive border-t pt-2">
                Error: {job.progress.error}
              </p>
            )}
          </div>
        ) : (
          <p className="text-sm text-muted-foreground">No progress data available</p>
        )}
        <div className="mt-3 pt-3 border-t text-xs text-muted-foreground">
          Created: {new Date(job.created_at).toLocaleString()}
        </div>
      </CardContent>
    </Card>
  )
}
