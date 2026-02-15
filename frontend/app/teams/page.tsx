'use client'

import { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import { Skeleton } from '@/components/ui/skeleton'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import { Users, RefreshCw, Search, Target, TrendingDown, TrendingUp, X } from 'lucide-react'
import { useTeams, useTeamForm } from '@/lib/queries/use-teams'
import type { Team, FormBreakdown } from '@/lib/types/api'

function FormIndicator({ value, label }: { value: number; label: string }) {
  const percentage = (value * 100).toFixed(1)
  const isGood = label === 'win_rate' || label === 'avg_goals_for'
  const isBad = label === 'loss_rate' || label === 'avg_goals_against'
  
  return (
    <div className="flex items-center justify-between p-2 bg-muted rounded">
      <span className="text-sm capitalize">{label.replace(/_/g, ' ')}</span>
      <div className="flex items-center gap-1">
        <span className="font-mono text-sm">{percentage}%</span>
        {isGood && value > 0.5 && <TrendingUp className="h-3 w-3 text-green-500" />}
        {isBad && value > 0.5 && <TrendingDown className="h-3 w-3 text-red-500" />}
      </div>
    </div>
  )
}

function TeamFormPanel({ team, onClose }: { team: Team; onClose: () => void }) {
  const { data: form, isLoading } = useTeamForm(team.id, 5)
  
  return (
    <Card className="mt-4">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg">{team.name} - Recent Form</CardTitle>
          <Button variant="ghost" size="sm" onClick={onClose}>
            <X className="h-4 w-4" />
          </Button>
        </div>
        <CardDescription>Performance over the last 5 matches</CardDescription>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <div className="space-y-2">
            {[...Array(6)].map((_, i) => (
              <Skeleton key={i} className="h-10" />
            ))}
          </div>
        ) : form ? (
          <div className="space-y-2">
            <FormIndicator value={form.win_rate} label="win_rate" />
            <FormIndicator value={form.draw_rate} label="draw_rate" />
            <FormIndicator value={form.loss_rate} label="loss_rate" />
            <div className="border-t pt-2 mt-2 space-y-2">
              <FormIndicator value={form.avg_goals_for} label="avg_goals_for" />
              <FormIndicator value={form.avg_goals_against} label="avg_goals_against" />
            </div>
            <div className="flex justify-center pt-4">
              <div className="text-center">
                <div className="text-3xl font-bold">{form.avg_points.toFixed(2)}</div>
                <div className="text-sm text-muted-foreground">Avg Points/Match</div>
              </div>
            </div>
          </div>
        ) : (
          <div className="text-center py-8 text-muted-foreground">
            No form data available
          </div>
        )}
      </CardContent>
    </Card>
  )
}

export default function TeamsPage() {
  const [search, setSearch] = useState('')
  const [debouncedSearch, setDebouncedSearch] = useState('')
  const [expandedTeam, setExpandedTeam] = useState<Team | null>(null)
  
  const { data: teams, isLoading, refetch } = useTeams({
    search: debouncedSearch || undefined,
    limit: 100,
  })

  const handleSearch = (value: string) => {
    setSearch(value)
    // Debounce search
    setTimeout(() => {
      setDebouncedSearch(value)
    }, 300)
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight flex items-center gap-2">
            <Users className="h-8 w-8" />
            Teams
          </h1>
          <p className="text-muted-foreground">
            Browse teams and view performance statistics
          </p>
        </div>
        <Button variant="outline" size="sm" onClick={() => refetch()}>
          <RefreshCw className="h-4 w-4 mr-2" />
          Refresh
        </Button>
      </div>

      {/* Search */}
      <Card>
        <CardContent className="p-4">
          <div className="flex items-center gap-4">
            <div className="relative flex-1 max-w-sm">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Search teams..."
                value={search}
                onChange={(e) => handleSearch(e.target.value)}
                className="pl-9"
              />
            </div>
            {debouncedSearch && (
              <Badge variant="secondary">
                {teams?.length ?? 0} results for "{debouncedSearch}"
              </Badge>
            )}
          </div>
        </CardContent>
      </Card>

      {isLoading ? (
        <div className="space-y-4">
          {[...Array(10)].map((_, i) => (
            <Skeleton key={i} className="h-16" />
          ))}
        </div>
      ) : teams && teams.length > 0 ? (
        <Card>
          <CardHeader>
            <CardTitle>All Teams</CardTitle>
            <CardDescription>
              {teams.length} teams in the database
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="w-16">ID</TableHead>
                  <TableHead>Team Name</TableHead>
                  <TableHead className="text-right">Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {teams.map((team) => (
                  <>
                    <TableRow key={team.id}>
                      <TableCell className="font-mono text-muted-foreground">
                        {team.id}
                      </TableCell>
                      <TableCell className="font-medium">
                        {team.name}
                      </TableCell>
                      <TableCell className="text-right">
                        <Button 
                          variant="outline" 
                          size="sm"
                          onClick={() => setExpandedTeam(expandedTeam?.id === team.id ? null : team)}
                        >
                          <Target className="h-4 w-4 mr-1" />
                          Form
                        </Button>
                      </TableCell>
                    </TableRow>
                    {expandedTeam?.id === team.id && (
                      <TableRow>
                        <TableCell colSpan={3} className="p-0">
                          <TeamFormPanel team={team} onClose={() => setExpandedTeam(null)} />
                        </TableCell>
                      </TableRow>
                    )}
                  </>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      ) : (
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-12 text-muted-foreground">
            <Users className="h-12 w-12 mb-4" />
            <p className="text-lg font-medium">No teams found</p>
            <p className="text-sm">
              {debouncedSearch 
                ? 'Try a different search term'
                : 'Scrape match data to populate teams'}
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  )
}