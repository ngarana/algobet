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
    onProgress: (progress: ScrapingProgressType) => {
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
      const fetchedJobs = await getScrapingJobs()
      setJobs(fetchedJobs)
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

  // Use WebSocket progress if available, otherwise fall back to job progress
  const displayProgress = currentProgress || currentJob?.progress || null

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
            url={currentJob.url}
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
          jobs={jobs.map((job) => ({
            id: job.id,
            job_type: job.job_type,
            url: job.url,
            status: job.status,
            created_at: job.created_at,
            progress: job.progress
              ? {
                  current_page: job.progress.current_page,
                  total_pages: job.progress.total_pages,
                  matches_scraped: job.progress.matches_scraped,
                  matches_saved: job.progress.matches_saved,
                }
              : null,
          }))}
          isLoading={isLoading}
          onRefresh={refreshJobs}
        />
      </div>
    </div>
  )
}