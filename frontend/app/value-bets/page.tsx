'use client'

import { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Skeleton } from '@/components/ui/skeleton'
import { Badge } from '@/components/ui/badge'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import { TrendingUp, Filter, RefreshCw, Trophy, Calendar, Target } from 'lucide-react'
import { useValueBets } from '@/lib/queries/use-value-bets'
import { useActiveModel } from '@/lib/queries/use-models'
import type { ValueBet } from '@/lib/types/api'

function ValueBetsFilters({
  filters,
  onFilterChange,
}: {
  filters: {
    minEv: number
    maxOdds: number
    days: number
  }
  onFilterChange: (filters: { minEv: number; maxOdds: number; days: number }) => void
}) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-lg flex items-center gap-2">
          <Filter className="h-5 w-5" />
          Filters
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-3 gap-4">
          <div className="space-y-2">
            <Label htmlFor="min-ev">Min EV (%)</Label>
            <Input
              id="min-ev"
              type="number"
              step="0.01"
              value={filters.minEv * 100}
              onChange={(e) =>
                onFilterChange({ ...filters, minEv: parseFloat(e.target.value) / 100 || 0 })
              }
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="max-odds">Max Odds</Label>
            <Input
              id="max-odds"
              type="number"
              step="0.1"
              value={filters.maxOdds}
              onChange={(e) =>
                onFilterChange({ ...filters, maxOdds: parseFloat(e.target.value) || 10 })
              }
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="days">Days Ahead</Label>
            <Select
              value={filters.days.toString()}
              onValueChange={(v) => onFilterChange({ ...filters, days: parseInt(v) })}
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="1">1 day</SelectItem>
                <SelectItem value="3">3 days</SelectItem>
                <SelectItem value="7">7 days</SelectItem>
                <SelectItem value="14">14 days</SelectItem>
                <SelectItem value="30">30 days</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

function ValueBetCard({ valueBet }: { valueBet: ValueBet }) {
  const outcomeLabels: Record<string, string> = {
    H: 'Home Win',
    D: 'Draw',
    A: 'Away Win',
  }

  const outcomeColors: Record<string, string> = {
    H: 'bg-blue-500',
    D: 'bg-yellow-500',
    A: 'bg-red-500',
  }

  const matchDate = new Date(valueBet.match.match_date)
  const ev = valueBet.expected_value

  return (
    <Card className="hover:shadow-md transition-shadow">
      <CardContent className="p-4">
        <div className="flex items-start justify-between mb-3">
          <div className="flex items-center gap-2">
            <Badge className={outcomeColors[valueBet.predicted_outcome]}>
              {outcomeLabels[valueBet.predicted_outcome]}
            </Badge>
            <span className="text-lg font-bold">@ {valueBet.market_odds.toFixed(2)}</span>
          </div>
          <Badge variant={ev >= 0.1 ? 'default' : 'secondary'}>
            EV: {(ev * 100).toFixed(1)}%
          </Badge>
        </div>

        <div className="space-y-2">
          <div className="font-medium">
            {valueBet.match.home_team_id} vs {valueBet.match.away_team_id}
          </div>

          <div className="flex items-center gap-4 text-sm text-muted-foreground">
            <span className="flex items-center gap-1">
              <Calendar className="h-3 w-3" />
              {matchDate.toLocaleDateString()}
            </span>
            <span className="flex items-center gap-1">
              <Target className="h-3 w-3" />
              {(valueBet.predicted_probability * 100).toFixed(1)}% prob
            </span>
          </div>

          <div className="flex items-center justify-between pt-2 border-t">
            <div className="text-sm">
              <span className="text-muted-foreground">Kelly:</span>{' '}
              <span className="font-mono">{(valueBet.kelly_fraction * 100).toFixed(1)}%</span>
            </div>
            <div className="text-sm">
              <span className="text-muted-foreground">Confidence:</span>{' '}
              <span className="font-mono">{(valueBet.confidence * 100).toFixed(1)}%</span>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

function ValueBetsTable({ valueBets }: { valueBets: ValueBet[] }) {
  const outcomeLabels: Record<string, string> = {
    H: 'Home Win',
    D: 'Draw',
    A: 'Away Win',
  }

  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Match</TableHead>
          <TableHead>Date</TableHead>
          <TableHead>Outcome</TableHead>
          <TableHead className="text-right">Odds</TableHead>
          <TableHead className="text-right">Prob</TableHead>
          <TableHead className="text-right">EV</TableHead>
          <TableHead className="text-right">Kelly</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {valueBets.map((vb, index) => (
          <TableRow key={`${vb.match.id}-${vb.predicted_outcome}-${index}`}>
            <TableCell className="font-medium">
              Team {vb.match.home_team_id} vs Team {vb.match.away_team_id}
            </TableCell>
            <TableCell>
              {new Date(vb.match.match_date).toLocaleDateString()}
            </TableCell>
            <TableCell>
              <Badge variant="outline">{outcomeLabels[vb.predicted_outcome]}</Badge>
            </TableCell>
            <TableCell className="text-right font-mono">
              {vb.market_odds.toFixed(2)}
            </TableCell>
            <TableCell className="text-right font-mono">
              {(vb.predicted_probability * 100).toFixed(1)}%
            </TableCell>
            <TableCell className="text-right font-mono text-green-600">
              +{(vb.expected_value * 100).toFixed(1)}%
            </TableCell>
            <TableCell className="text-right font-mono">
              {(vb.kelly_fraction * 100).toFixed(1)}%
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  )
}

function ValueBetsSummary({ valueBets }: { valueBets: ValueBet[] }) {
  const avgEv = valueBets.length > 0
    ? valueBets.reduce((sum, vb) => sum + vb.expected_value, 0) / valueBets.length
    : 0

  const avgOdds = valueBets.length > 0
    ? valueBets.reduce((sum, vb) => sum + vb.market_odds, 0) / valueBets.length
    : 0

  const avgKelly = valueBets.length > 0
    ? valueBets.reduce((sum, vb) => sum + vb.kelly_fraction, 0) / valueBets.length
    : 0

  return (
    <div className="grid grid-cols-4 gap-4">
      <Card>
        <CardContent className="p-4 text-center">
          <div className="text-2xl font-bold">{valueBets.length}</div>
          <div className="text-xs text-muted-foreground">Opportunities</div>
        </CardContent>
      </Card>
      <Card>
        <CardContent className="p-4 text-center">
          <div className="text-2xl font-bold text-green-600">
            +{(avgEv * 100).toFixed(1)}%
          </div>
          <div className="text-xs text-muted-foreground">Avg EV</div>
        </CardContent>
      </Card>
      <Card>
        <CardContent className="p-4 text-center">
          <div className="text-2xl font-bold">{avgOdds.toFixed(2)}</div>
          <div className="text-xs text-muted-foreground">Avg Odds</div>
        </CardContent>
      </Card>
      <Card>
        <CardContent className="p-4 text-center">
          <div className="text-2xl font-bold">{(avgKelly * 100).toFixed(1)}%</div>
          <div className="text-xs text-muted-foreground">Avg Kelly</div>
        </CardContent>
      </Card>
    </div>
  )
}

export default function ValueBetsPage() {
  const { data: activeModel } = useActiveModel()
  const [filters, setFilters] = useState({
    minEv: 0.05,
    maxOdds: 10,
    days: 7,
  })

  const { data, isLoading, refetch } = useValueBets({
    min_ev: filters.minEv,
    max_odds: filters.maxOdds,
    days: filters.days,
    max_matches: 50,
  })

  const valueBets = data ?? []

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight flex items-center gap-2">
            <TrendingUp className="h-8 w-8" />
            Value Bets
          </h1>
          <p className="text-muted-foreground">
            Find betting opportunities with positive expected value
          </p>
        </div>
        <div className="flex items-center gap-2">
          {activeModel && (
            <Badge variant="secondary" className="flex items-center gap-1">
              <Trophy className="h-3 w-3" />
              {activeModel.algorithm}
            </Badge>
          )}
          <Button variant="outline" size="sm" onClick={() => refetch()}>
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh
          </Button>
        </div>
      </div>

      <ValueBetsFilters filters={filters} onFilterChange={setFilters} />

      {isLoading ? (
        <div className="space-y-4">
          <div className="grid grid-cols-4 gap-4">
            {[...Array(4)].map((_, i) => (
              <Skeleton key={i} className="h-20" />
            ))}
          </div>
          <Skeleton className="h-64" />
        </div>
      ) : (
        <>
          <ValueBetsSummary valueBets={valueBets} />

          {valueBets.length > 0 ? (
            <Card>
              <CardHeader>
                <CardTitle>Value Betting Opportunities</CardTitle>
                <CardDescription>
                  {valueBets.length} matches with positive expected value
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ValueBetsTable valueBets={valueBets} />
              </CardContent>
            </Card>
          ) : (
            <Card>
              <CardContent className="flex flex-col items-center justify-center py-12 text-muted-foreground">
                <TrendingUp className="h-12 w-12 mb-4" />
                <p className="text-lg font-medium">No value bets found</p>
                <p className="text-sm">
                  Try adjusting your filters or wait for new matches
                </p>
              </CardContent>
            </Card>
          )}
        </>
      )}
    </div>
  )
}
