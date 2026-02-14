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
  job_type: string
  url: string
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'
  created_at: string
  progress: {
    current_page: number
    total_pages: number
    matches_scraped: number
    matches_saved: number
  } | null
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
                  <Badge variant={job.job_type === 'upcoming' ? 'default' : 'secondary'}>
                    {job.job_type}
                  </Badge>
                </TableCell>
                <TableCell>{getStatusBadge(job.status)}</TableCell>
                <TableCell className="max-w-xs truncate">
                  <a
                    href={job.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-blue-600 hover:underline"
                  >
                    {job.url}
                  </a>
                </TableCell>
                <TableCell className="text-sm text-muted-foreground">
                  {new Date(job.created_at).toLocaleString()}
                </TableCell>
                <TableCell>
                  {job.progress ? (
                    <div className="space-y-1 min-w-32">
                      <div className="flex justify-between text-xs">
                        <span>S: {job.progress.matches_scraped}</span>
                        <span>V: {job.progress.matches_saved}</span>
                      </div>
                      {job.progress.total_pages > 0 && (
                        <Progress className="h-2">
                          <ProgressValue
                            value={
                              (job.progress.current_page / job.progress.total_pages) * 100
                            }
                          />
                        </Progress>
                      )}
                    </div>
                  ) : (
                    <span className="text-muted-foreground text-sm">N/A</span>
                  )}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      )}
    </div>
  )
}
