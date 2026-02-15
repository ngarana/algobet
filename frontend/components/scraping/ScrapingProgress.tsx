'use client'

import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Progress, ProgressValue } from '@/components/ui/progress'
import {
  CheckCircleIcon,
  XCircleIcon,
  ClockIcon,
  Loader2Icon,
} from 'lucide-react'

export interface ScrapingProgressData {
  job_id: string
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'
  progress: number
  matches_scraped: number
  message: string
  error?: string | null
  started_at?: string | null
  completed_at?: string | null
}

interface ScrapingProgressProps {
  progress: ScrapingProgressData | null
  url?: string
}

function getStatusIcon(status: string) {
  switch (status) {
    case 'pending':
      return <ClockIcon className="w-5 h-5 text-muted-foreground" />
    case 'running':
      return <Loader2Icon className="w-5 h-5 text-primary animate-spin" />
    case 'completed':
      return <CheckCircleIcon className="w-5 h-5 text-green-500" />
    case 'failed':
      return <XCircleIcon className="w-5 h-5 text-destructive" />
    default:
      return <ClockIcon className="w-5 h-5 text-muted-foreground" />
  }
}

export function ScrapingProgress({ progress, url }: ScrapingProgressProps) {
  if (!progress) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Scraping Progress</CardTitle>
          <CardDescription>Waiting for scraping to start...</CardDescription>
        </CardHeader>
      </Card>
    )
  }


  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          {getStatusIcon(progress.status)}
          <div>
            <CardTitle>Scraping Progress</CardTitle>
            {url && <CardDescription className="truncate max-w-md">{url}</CardDescription>}
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-center justify-between">
          <Badge variant={progress.status === 'running' ? 'default' : 'secondary'}>
            {progress.status.toUpperCase()}
          </Badge>
          <span className="text-sm text-muted-foreground">
            Job ID: {progress.job_id}
          </span>
        </div>

        <div className="space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span>Overall Progress</span>
            <span className="font-medium">{progress.progress.toFixed(1)}%</span>
          </div>
          <Progress>
            <ProgressValue value={progress.progress} />
          </Progress>
        </div>

        <div className="grid grid-cols-1 gap-4">
          <div className="bg-muted/50 rounded-lg p-3">
            <p className="text-sm text-muted-foreground">Matches Scraped</p>
            <p className="text-2xl font-bold">{progress.matches_scraped}</p>
          </div>
        </div>

        {progress.message && (
          <div className="bg-muted/30 rounded-lg p-3">
            <p className="text-sm">{progress.message}</p>
          </div>
        )}

        {progress.error && (
          <div className="bg-destructive/10 text-destructive rounded-lg p-3">
            <p className="text-sm font-medium">Error</p>
            <p className="text-sm">{progress.error}</p>
          </div>
        )}

        <div className="text-xs text-muted-foreground space-y-1">
          {progress.started_at && (
            <p>Started: {new Date(progress.started_at).toLocaleString()}</p>
          )}
          {progress.completed_at && (
            <p>Completed: {new Date(progress.completed_at).toLocaleString()}</p>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
