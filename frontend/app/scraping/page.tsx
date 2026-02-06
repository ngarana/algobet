'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow
} from '@/components/ui/table';
import {
  PlayIcon,
  PauseIcon,
  RotateCcwIcon,
  ClockIcon,
  CheckCircleIcon,
  XCircleIcon,
  Loader2Icon
} from 'lucide-react';

// Mock API functions - to be replaced with actual API calls
const mockScrapeUpcoming = async (url: string) => {
  // Simulate API call
  return new Promise((resolve) => {
    setTimeout(() => {
      resolve({
        id: 'job-' + Date.now(),
        job_type: 'upcoming',
        url,
        status: 'running',
        created_at: new Date().toISOString(),
        progress: {
          job_id: 'job-' + Date.now(),
          status: 'running',
          current_page: 0,
          total_pages: 0,
          matches_scraped: 0,
          matches_saved: 0,
          message: 'Starting scrape...',
          error: null,
          started_at: new Date().toISOString(),
          completed_at: null
        }
      });
    }, 500);
  });
};

const mockGetJobs = async () => {
  // Simulate API call
  return new Promise((resolve) => {
    setTimeout(() => {
      resolve([
        {
          id: 'job-1',
          job_type: 'upcoming',
          url: 'https://www.oddsportal.com/matches/football/',
          status: 'completed',
          created_at: new Date(Date.now() - 3600000).toISOString(),
          progress: {
            job_id: 'job-1',
            status: 'completed',
            current_page: 1,
            total_pages: 1,
            matches_scraped: 45,
            matches_saved: 45,
            message: 'Completed! Scraped 45 matches, saved 45.',
            error: null,
            started_at: new Date(Date.now() - 3600000).toISOString(),
            completed_at: new Date(Date.now() - 1800000).toISOString()
          }
        },
        {
          id: 'job-2',
          job_type: 'results',
          url: 'https://www.oddsportal.com/football/england/premier-league/results/',
          status: 'running',
          created_at: new Date().toISOString(),
          progress: {
            job_id: 'job-2',
            status: 'running',
            current_page: 3,
            total_pages: 10,
            matches_scraped: 120,
            matches_saved: 118,
            message: 'Scraping page 3/10...',
            error: null,
            started_at: new Date().toISOString(),
            completed_at: null
          }
        }
      ]);
    }, 300);
  });
};

export default function ScrapingPage() {
  const [activeTab, setActiveTab] = useState('upcoming');
  const [upcomingUrl, setUpcomingUrl] = useState('https://www.oddsportal.com/matches/football/');
  const [resultsUrl, setResultsUrl] = useState('');
  const [maxPages, setMaxPages] = useState<number | undefined>(undefined);
  const [jobs, setJobs] = useState<any[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [scrapingJob, setScrapingJob] = useState<any>(null);

  const startScraping = async (type: string) => {
    setIsLoading(true);
    try {
      if (type === 'upcoming') {
        const job = await mockScrapeUpcoming(upcomingUrl);
        setScrapingJob(job);
      } else if (type === 'results') {
        // For results, we'd call the appropriate API
        console.log('Starting results scrape with URL:', resultsUrl, 'Max pages:', maxPages);
      }
    } catch (error) {
      console.error('Error starting scrape:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const refreshJobs = async () => {
    setIsLoading(true);
    try {
      const fetchedJobs = await mockGetJobs();
      setJobs(fetchedJobs as any[]);
    } catch (error) {
      console.error('Error fetching jobs:', error);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    refreshJobs();
  }, []);

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'pending':
        return <Badge variant="secondary"><ClockIcon className="w-3 h-3 mr-1" /> Pending</Badge>;
      case 'running':
        return <Badge variant="default"><Loader2Icon className="w-3 h-3 mr-1 animate-spin" /> Running</Badge>;
      case 'completed':
        return <Badge variant="success"><CheckCircleIcon className="w-3 h-3 mr-1" /> Completed</Badge>;
      case 'failed':
        return <Badge variant="destructive"><XCircleIcon className="w-3 h-3 mr-1" /> Failed</Badge>;
      default:
        return <Badge variant="outline">{status}</Badge>;
    }
  };

  return (
    <div className="container mx-auto py-10">
      <div className="mb-8">
        <h1 className="text-3xl font-bold tracking-tight">Scraping Management</h1>
        <p className="text-muted-foreground mt-2">
          Control data scraping operations and monitor progress in real-time
        </p>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="upcoming">Upcoming Matches</TabsTrigger>
          <TabsTrigger value="results">Historical Results</TabsTrigger>
        </TabsList>

        <TabsContent value="upcoming" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Scrape Upcoming Matches</CardTitle>
              <CardDescription>
                Fetch upcoming matches with odds from OddsPortal
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="upcoming-url">OddsPortal URL</Label>
                <Input
                  id="upcoming-url"
                  value={upcomingUrl}
                  onChange={(e) => setUpcomingUrl(e.target.value)}
                  placeholder="https://www.oddsportal.com/matches/football/"
                />
              </div>
              <Button
                onClick={() => startScraping('upcoming')}
                disabled={isLoading}
                className="w-full"
              >
                {isLoading ? (
                  <>
                    <Loader2Icon className="mr-2 h-4 w-4 animate-spin" />
                    Starting...
                  </>
                ) : (
                  <>
                    <PlayIcon className="mr-2 h-4 w-4" />
                    Start Scraping
                  </>
                )}
              </Button>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="results" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Scrape Historical Results</CardTitle>
              <CardDescription>
                Fetch historical match results from OddsPortal
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="results-url">OddsPortal Results URL</Label>
                <Input
                  id="results-url"
                  value={resultsUrl}
                  onChange={(e) => setResultsUrl(e.target.value)}
                  placeholder="https://www.oddsportal.com/football/england/premier-league/results/"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="max-pages">Max Pages (optional)</Label>
                <Input
                  id="max-pages"
                  type="number"
                  value={maxPages || ''}
                  onChange={(e) => setMaxPages(e.target.value ? parseInt(e.target.value) : undefined)}
                  placeholder="Leave blank for all pages"
                />
              </div>
              <Button
                onClick={() => startScraping('results')}
                disabled={isLoading || !resultsUrl}
                className="w-full"
              >
                {isLoading ? (
                  <>
                    <Loader2Icon className="mr-2 h-4 w-4 animate-spin" />
                    Starting...
                  </>
                ) : (
                  <>
                    <PlayIcon className="mr-2 h-4 w-4" />
                    Start Scraping
                  </>
                )}
              </Button>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Current Scraping Job Progress */}
      {scrapingJob && (
        <Card className="mt-6">
          <CardHeader>
            <CardTitle>Current Job Progress</CardTitle>
            <CardDescription>
              {scrapingJob.url}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Status</span>
                {getStatusBadge(scrapingJob.status)}
              </div>
              {scrapingJob.progress && (
                <>
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium">Progress</span>
                      <span className="text-sm text-muted-foreground">
                        {scrapingJob.progress.current_page} / {scrapingJob.progress.total_pages || '?'}
                      </span>
                    </div>
                    <Progress value={scrapingJob.progress.total_pages ?
                      (scrapingJob.progress.current_page / scrapingJob.progress.total_pages) * 100 : 0}
                    />
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <p className="text-sm text-muted-foreground">Matches Scraped</p>
                      <p className="text-lg font-semibold">{scrapingJob.progress.matches_scraped}</p>
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground">Matches Saved</p>
                      <p className="text-lg font-semibold">{scrapingJob.progress.matches_saved}</p>
                    </div>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Message</p>
                    <p className="text-sm">{scrapingJob.progress.message}</p>
                  </div>
                </>
              )}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Job History */}
      <Card className="mt-6">
        <CardHeader>
          <div className="flex justify-between items-center">
            <div>
              <CardTitle>Scraping Jobs History</CardTitle>
              <CardDescription>
                Previous and ongoing scraping operations
              </CardDescription>
            </div>
            <Button variant="outline" size="sm" onClick={refreshJobs}>
              <RotateCcwIcon className="h-4 w-4 mr-2" />
              Refresh
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="flex justify-center items-center h-32">
              <Loader2Icon className="h-8 w-8 animate-spin" />
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
                    <TableCell className="font-medium">{job.id}</TableCell>
                    <TableCell>
                      <Badge variant={job.job_type === 'upcoming' ? 'default' : 'secondary'}>
                        {job.job_type}
                      </Badge>
                    </TableCell>
                    <TableCell>
                      {getStatusBadge(job.status)}
                    </TableCell>
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
                    <TableCell>
                      {new Date(job.created_at).toLocaleString()}
                    </TableCell>
                    <TableCell>
                      {job.progress && (
                        <div className="space-y-1">
                          <div className="flex justify-between text-xs">
                            <span>Scraped: {job.progress.matches_scraped}</span>
                            <span>Saved: {job.progress.matches_saved}</span>
                          </div>
                          {job.progress.total_pages > 0 && (
                            <Progress
                              value={job.progress.current_page / job.progress.total_pages * 100}
                              className="h-2"
                            />
                          )}
                        </div>
                      )}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
