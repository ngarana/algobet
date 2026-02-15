'use client'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import { Progress, ProgressValue } from '@/components/ui/progress'
import {
  CheckCircleIcon,
  XCircleIcon,
  ClockIcon,
  Loader2Icon,
  RotateCcwIcon,
} from 'lucide-react'

export interface JobHistoryItem {
  id: string
  scraping_type: string
  tournament_url: string | null
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'
  created_at: string
  progress: number
  matches_scraped: number
  message?: string | null
}

interface JobHistoryTableProps {
  jobs: JobHistoryItem[]
  isLoading?: boolean
  onRefresh?: () => void
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
    default:
      return <Badge variant="outline">{status}</Badge>
  }
}

export function JobHistoryTable({ jobs, isLoading = false, onRefresh }: JobHistoryTableProps) {
  return (
    <div className="rounded-md border">
      <div className="flex items-center justify-between p-4 border-b">
        <h3 className="font-semibold">Scraping Jobs History</h3>
        {onRefresh && (
          <Button variant="outline" size="sm" onClick={onRefresh} disabled={isLoading}>
            <RotateCcwIcon className="h-4 w-4 mr-2" />
            Refresh
          </Button>
        )}
      </div>
      {isLoading ? (
        <div className="flex justify-center items-center h-32">
          <Loader2Icon className="h-8 w-8 animate-spin text-muted-foreground" />
        </div>
      ) : jobs.length === 0 ? (
        <div className="flex justify-center items-center h-32 text-muted-foreground">
          No scraping jobs found
        </div>
      ) : (
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Job ID</TableHead>
              <TableHead>Type</TableHead>
              <TableHead>Status</TableHead>
              <TableHead>URL</TableHead>
              <TableHead>Created</TableHead>
              <TableHead>Progress</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {jobs.map((job) => (
              <TableRow key={job.id}>
                <TableCell className="font-medium font-mono text-xs">
                  {job.id.slice(0, 8)}...
                </TableCell>
                <TableCell>
                  <Badge variant={job.scraping_type === 'upcoming' ? 'default' : 'secondary'}>
                    {job.scraping_type}
                  </Badge>
                </TableCell>
                <TableCell>{getStatusBadge(job.status)}</TableCell>
                <TableCell className="max-w-xs truncate">
                  {job.tournament_url ? (
                    <a
                      href={job.tournament_url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-blue-600 hover:underline"
                    >
                      {job.tournament_url}
                    </a>
                  ) : (
                    <span className="text-muted-foreground text-sm">All</span>
                  )}
                </TableCell>
                <TableCell className="text-sm text-muted-foreground">
                  {new Date(job.created_at).toLocaleString()}
                </TableCell>
                <TableCell>
                  <div className="space-y-1 min-w-32">
                    <div className="flex justify-between text-xs">
                      <span>M: {job.matches_scraped}</span>
                      <span>{job.progress.toFixed(0)}%</span>
                    </div>
                    <Progress className="h-2">
                      <ProgressValue value={job.progress} />
                    </Progress>
                  </div>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      )}
    </div>
  )
}
