'use client'

import { useState, useEffect, useCallback } from 'react'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { AlertCircleIcon } from 'lucide-react'
import { ScrapeForm, JobHistoryTable, ScrapingProgressCard } from '@/components/scraping'
import { useScrapingProgress } from '@/hooks/useScrapingProgress'
import {
  scrapeUpcomingMatches,
  scrapeResults,
  getScrapingJobs,
  type ScrapingJob,
  type ScrapingProgress as ScrapingProgressType,
} from '@/lib/api/scraping'

export default function ScrapingPage() {
  const [activeTab, setActiveTab] = useState('upcoming')
  const [jobs, setJobs] = useState<ScrapingJob[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [currentJob, setCurrentJob] = useState<ScrapingJob | null>(null)
  const [error, setError] = useState<string | null>(null)

  // WebSocket connection for real-time progress
  const { currentProgress, isConnected } = useScrapingProgress({
    jobId: currentJob?.id,
    onProgress: (progress: any) => {
      if (progress.status === 'completed' || progress.status === 'failed') {
        // Refresh jobs list when job completes
        refreshJobs()
      }
    },
    onError: () => {
      setError('WebSocket connection error. Progress updates may be delayed.')
    },
  })

  const refreshJobs = useCallback(async () => {
    try {
      const response = await getScrapingJobs()
      setJobs(response.items)
    } catch (err) {
      console.error('Error fetching jobs:', err)
      setError('Failed to fetch scraping jobs')
    }
  }, [])

  useEffect(() => {
    const loadJobs = async () => {
      setIsLoading(true)
      try {
        await refreshJobs()
      } finally {
        setIsLoading(false)
      }
    }
    loadJobs()
  }, [refreshJobs])

  const handleScrapeUpcoming = async (data: { url: string }) => {
    setIsSubmitting(true)
    setError(null)
    try {
      const job = await scrapeUpcomingMatches({ url: data.url })
      setCurrentJob(job)
      setJobs((prev) => [job, ...prev])
    } catch (err) {
      console.error('Error starting upcoming scrape:', err)
      setError('Failed to start scraping. Please check the URL and try again.')
    } finally {
      setIsSubmitting(false)
    }
  }

  const handleScrapeResults = async (data: { url: string; maxPages?: number }) => {
    setIsSubmitting(true)
    setError(null)
    try {
      const job = await scrapeResults({
        url: data.url,
        max_pages: data.maxPages,
      })
      setCurrentJob(job)
      setJobs((prev) => [job, ...prev])
    } catch (err) {
      console.error('Error starting results scrape:', err)
      setError('Failed to start scraping. Please check the URL and try again.')
    } finally {
      setIsSubmitting(false)
    }
  }

  // Map display progress correctly
  const displayProgress: any = currentProgress ? {
    job_id: currentProgress.job_id,
    status: currentProgress.status || 'running',
    progress: currentProgress.progress,
    matches_scraped: currentProgress.matches_scraped,
    message: currentProgress.message,
    timestamp: currentProgress.timestamp,
  } : currentJob ? {
    job_id: currentJob.id,
    status: currentJob.status,
    progress: currentJob.progress,
    matches_scraped: currentJob.matches_scraped,
    message: currentJob.message,
    started_at: currentJob.started_at,
    completed_at: currentJob.completed_at,
  } : null

  return (
    <div className="container mx-auto py-10">
      <div className="mb-8">
        <h1 className="text-3xl font-bold tracking-tight">Scraping Management</h1>
        <p className="text-muted-foreground mt-2">
          Control data scraping operations and monitor progress in real-time
        </p>
      </div>

      {error && (
        <Card className="mb-6 border-destructive">
          <CardContent className="flex items-center gap-2 p-4 text-destructive">
            <AlertCircleIcon className="h-5 w-5" />
            <p>{error}</p>
          </CardContent>
        </Card>
      )}

      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="upcoming">Upcoming Matches</TabsTrigger>
          <TabsTrigger value="results">Historical Results</TabsTrigger>
        </TabsList>

        <TabsContent value="upcoming" className="space-y-6">
          <ScrapeForm
            type="upcoming"
            onSubmit={handleScrapeUpcoming}
            isLoading={isSubmitting}
            defaultUrl="https://www.oddsportal.com/matches/football/"
          />
        </TabsContent>

        <TabsContent value="results" className="space-y-6">
          <ScrapeForm
            type="results"
            onSubmit={handleScrapeResults}
            isLoading={isSubmitting}
          />
        </TabsContent>
      </Tabs>

      {/* Current Scraping Job Progress */}
      {currentJob && (
        <div className="mt-6">
          <ScrapingProgressCard
            progress={displayProgress}
            url={currentJob.tournament_url || undefined}
          />
          {isConnected && currentJob && (
            <p className="text-xs text-muted-foreground mt-2">
              ● Connected for real-time updates
            </p>
          )}
        </div>
      )}

      {/* Job History */}
      <div className="mt-6">
        <JobHistoryTable
          jobs={(jobs as any)?.items?.map((job: any) => ({
            id: job.id,
            scraping_type: job.scraping_type,
            tournament_url: job.tournament_url,
            status: job.status,
            created_at: job.created_at,
            progress: job.progress,
            matches_scraped: job.matches_scraped,
            message: job.message,
          })) || []}
          isLoading={isLoading}
          onRefresh={refreshJobs}
        />
      </div>
    </div>
  )
}