'use client'

import { useParams } from 'next/navigation'
import { format } from 'date-fns'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Skeleton } from '@/components/ui/skeleton'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { useMatch, useMatchH2H } from '@/lib/queries/use-matches'
import type { MatchStatus } from '@/lib/types/api'

function getStatusColor(status: MatchStatus): string {
  switch (status) {
    case 'LIVE':
      return 'bg-red-500 text-white animate-pulse'
    case 'FINISHED':
      return 'bg-muted text-muted-foreground'
    case 'SCHEDULED':
      return 'bg-blue-500 text-white'
    default:
      return 'bg-muted text-muted-foreground'
  }
}

function MatchInfo() {
  const params = useParams()
  const matchId = parseInt(params.id as string)
  const { data: match, isLoading } = useMatch(matchId)

  if (isLoading) {
    return (
      <Card>
        <CardContent className="p-6">
          <div className="flex flex-col items-center space-y-4">
            <Skeleton className="h-8 w-48" />
            <Skeleton className="h-6 w-32" />
            <div className="flex items-center justify-center gap-8 w-full">
              <div className="text-center space-y-2">
                <Skeleton className="h-12 w-32" />
                <Skeleton className="h-4 w-24" />
              </div>
              <Skeleton className="h-16 w-24" />
              <div className="text-center space-y-2">
                <Skeleton className="h-12 w-32" />
                <Skeleton className="h-4 w-24" />
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    )
  }

  if (!match) {
    return (
      <Card>
        <CardContent className="p-6">
          <p className="text-center text-muted-foreground">Match not found</p>
        </CardContent>
      </Card>
    )
  }

  const isFinished = match.status === 'FINISHED'
  const isLive = match.status === 'LIVE'

  return (
    <Card>
      <CardContent className="p-6">
        <div className="flex flex-col items-center space-y-4">
          <Badge className={getStatusColor(match.status)}>
            {match.status}
          </Badge>
          
          <p className="text-sm text-muted-foreground">
            {format(new Date(match.match_date), 'EEEE, MMMM d, yyyy HH:mm')}
          </p>

          <div className="flex items-center justify-center gap-8 w-full">
            {/* Home Team */}
            <div className="text-center flex-1">
              <h2 className="text-2xl font-bold">{match.home_team?.name || `Team ${match.home_team_id}`}</h2>
              <p className="text-sm text-muted-foreground">Home</p>
            </div>

            {/* Score */}
            <div className="text-center px-8">
              {isFinished || isLive ? (
                <div className="text-5xl font-bold">
                  {match.home_score} - {match.away_score}
                </div>
              ) : (
                <div className="text-3xl font-bold text-muted-foreground">VS</div>
              )}
            </div>

            {/* Away Team */}
            <div className="text-center flex-1">
              <h2 className="text-2xl font-bold">{match.away_team?.name || `Team ${match.away_team_id}`}</h2>
              <p className="text-sm text-muted-foreground">Away</p>
            </div>
          </div>

          {/* Odds */}
          {match.odds_home && match.odds_draw && match.odds_away && (
            <div className="flex items-center gap-6 pt-4 border-t w-full justify-center">
              <div className="text-center">
                <p className="text-sm text-muted-foreground">Home Win</p>
                <p className="text-xl font-bold">{match.odds_home.toFixed(2)}</p>
              </div>
              <div className="text-center">
                <p className="text-sm text-muted-foreground">Draw</p>
                <p className="text-xl font-bold">{match.odds_draw.toFixed(2)}</p>
              </div>
              <div className="text-center">
                <p className="text-sm text-muted-foreground">Away Win</p>
                <p className="text-xl font-bold">{match.away_score?.toFixed(2) || match.odds_away.toFixed(2)}</p>
              </div>
            </div>
          )}

          {/* Tournament Info */}
          {match.tournament && (
            <div className="pt-4 text-center">
              <p className="text-sm text-muted-foreground">
                {match.tournament.name} • {match.season?.name}
              </p>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )
}

function H2HTable() {
  const params = useParams()
  const matchId = parseInt(params.id as string)
  const { data, isLoading } = useMatchH2H(matchId)

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Head to Head</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            {Array.from({ length: 5 }).map((_, i) => (
              <Skeleton key={i} className="h-12 w-full" />
            ))}
          </div>
        </CardContent>
      </Card>
    )
  }

  const h2hMatches = data?.items?.slice(0, 5) || []

  return (
    <Card>
      <CardHeader>
        <CardTitle>Head to Head (Last 5)</CardTitle>
      </CardHeader>
      <CardContent>
        {h2hMatches.length === 0 ? (
          <p className="text-center text-muted-foreground py-4">
            No previous meetings found
          </p>
        ) : (
          <div className="space-y-2">
            {h2hMatches.map((match) => (
              <div
                key={match.id}
                className="flex items-center justify-between p-3 border rounded-lg"
              >
                <span className="text-sm text-muted-foreground">
                  {format(new Date(match.match_date), 'MMM d, yyyy')}
                </span>
                <div className="flex items-center gap-2">
                  <span className="font-medium">{match.home_score}</span>
                  <span className="text-muted-foreground">-</span>
                  <span className="font-medium">{match.away_score}</span>
                </div>
                <Badge variant={match.result === 'H' ? 'default' : 'secondary'}>
                  {match.result === 'H' ? 'H' : match.result === 'A' ? 'A' : 'D'}
                </Badge>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  )
}

function PredictionsCard() {
  const params = useParams()
  const matchId = parseInt(params.id as string)
  const { data: match, isLoading } = useMatch(matchId)

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>AI Prediction</CardTitle>
        </CardHeader>
        <CardContent>
          <Skeleton className="h-32 w-full" />
        </CardContent>
      </Card>
    )
  }

  const predictions = match?.predictions || []
  
  if (predictions.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>AI Prediction</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-center text-muted-foreground py-4">
            No predictions available for this match
          </p>
        </CardContent>
      </Card>
    )
  }

  const latestPrediction = predictions[0]

  return (
    <Card>
      <CardHeader>
        <CardTitle>AI Prediction</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {/* Probability Bars */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-sm">Home Win</span>
              <span className="font-bold">{(latestPrediction.prob_home * 100).toFixed(1)}%</span>
            </div>
            <div className="h-2 bg-muted rounded-full overflow-hidden">
              <div 
                className="h-full bg-blue-500 rounded-full"
                style={{ width: `${latestPrediction.prob_home * 100}%` }}
              />
            </div>

            <div className="flex items-center justify-between">
              <span className="text-sm">Draw</span>
              <span className="font-bold">{(latestPrediction.prob_draw * 100).toFixed(1)}%</span>
            </div>
            <div className="h-2 bg-muted rounded-full overflow-hidden">
              <div 
                className="h-full bg-yellow-500 rounded-full"
                style={{ width: `${latestPrediction.prob_draw * 100}%` }}
              />
            </div>

            <div className="flex items-center justify-between">
              <span className="text-sm">Away Win</span>
              <span className="font-bold">{(latestPrediction.prob_away * 100).toFixed(1)}%</span>
            </div>
            <div className="h-2 bg-muted rounded-full overflow-hidden">
              <div 
                className="h-full bg-red-500 rounded-full"
                style={{ width: `${latestPrediction.prob_away * 100}%` }}
              />
            </div>
          </div>

          {/* Prediction Result */}
          <div className="pt-4 border-t">
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground">Predicted Outcome</span>
              <Badge 
                variant={latestPrediction.predicted_outcome === 'H' ? 'default' : 'secondary'}
                className="text-lg"
              >
                {latestPrediction.predicted_outcome === 'H' ? 'Home Win' : 
                 latestPrediction.predicted_outcome === 'A' ? 'Away Win' : 'Draw'}
              </Badge>
            </div>
            <div className="flex items-center justify-between mt-2">
              <span className="text-muted-foreground">Confidence</span>
              <span className="font-bold">{(latestPrediction.confidence * 100).toFixed(1)}%</span>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

export default function MatchDetailPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Match Details</h1>
        <p className="text-muted-foreground">
          View match information, statistics, and predictions
        </p>
      </div>

      <MatchInfo />

      <Tabs defaultValue="h2h" className="w-full">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="h2h">Head to Head</TabsTrigger>
          <TabsTrigger value="predictions">Predictions</TabsTrigger>
          <TabsTrigger value="stats">Statistics</TabsTrigger>
        </TabsList>
        
        <TabsContent value="h2h" className="mt-4">
          <H2HTable />
        </TabsContent>
        
        <TabsContent value="predictions" className="mt-4">
          <PredictionsCard />
        </TabsContent>
        
        <TabsContent value="stats" className="mt-4">
          <Card>
            <CardHeader>
              <CardTitle>Team Statistics</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-center text-muted-foreground py-4">
                Team form charts and statistics will be displayed here
              </p>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}
