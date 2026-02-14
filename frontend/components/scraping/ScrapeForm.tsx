'use client'

import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { PlayIcon, Loader2Icon } from 'lucide-react'

interface ScrapeFormProps {
  type: 'upcoming' | 'results'
  onSubmit: (data: { url: string; maxPages?: number }) => Promise<void>
  isLoading?: boolean
  defaultUrl?: string
}

export function ScrapeForm({ type, onSubmit, isLoading = false, defaultUrl }: ScrapeFormProps) {
  const [url, setUrl] = useState(
    defaultUrl ||
      (type === 'upcoming'
        ? 'https://www.oddsportal.com/matches/football/'
        : '')
  )
  const [maxPages, setMaxPages] = useState<number | undefined>(undefined)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!url) return
    await onSubmit({ url, maxPages })
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>
          {type === 'upcoming' ? 'Scrape Upcoming Matches' : 'Scrape Historical Results'}
        </CardTitle>
        <CardDescription>
          {type === 'upcoming'
            ? 'Fetch upcoming matches with odds from OddsPortal'
            : 'Fetch historical match results from OddsPortal'}
        </CardDescription>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor={`${type}-url`}>OddsPortal URL</Label>
            <Input
              id={`${type}-url`}
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder={
                type === 'upcoming'
                  ? 'https://www.oddsportal.com/matches/football/'
                  : 'https://www.oddsportal.com/football/england/premier-league/results/'
              }
              required
            />
          </div>

          {type === 'results' && (
            <div className="space-y-2">
              <Label htmlFor="max-pages">Max Pages (optional)</Label>
              <Input
                id="max-pages"
                type="number"
                min={1}
                value={maxPages ?? ''}
                onChange={(e) =>
                  setMaxPages(e.target.value ? parseInt(e.target.value) : undefined)
                }
                placeholder="Leave blank for all pages"
              />
            </div>
          )}

          <Button type="submit" disabled={isLoading || !url} className="w-full">
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
        </form>
      </CardContent>
    </Card>
  )
}
