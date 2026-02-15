/**
 * TanStack Query hooks for model versions
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  getModels,
  getActiveModel,
  getModel,
  activateModel,
  deleteModel,
  getModelMetrics,
} from '@/lib/api/models'

export const modelKeys = {
  all: ['models'] as const,
  lists: () => [...modelKeys.all, 'list'] as const,
  active: () => [...modelKeys.all, 'active'] as const,
  detail: (id: number) => [...modelKeys.all, 'detail', id] as const,
  metrics: (id: number) => [...modelKeys.detail(id), 'metrics'] as const,
}

export function useModels() {
  return useQuery({
    queryKey: modelKeys.lists(),
    queryFn: getModels,
  })
}

export function useActiveModel() {
  return useQuery({
    queryKey: modelKeys.active(),
    queryFn: getActiveModel,
  })
}

export function useModel(id: number) {
  return useQuery({
    queryKey: modelKeys.detail(id),
    queryFn: () => getModel(id),
    enabled: !!id,
  })
}

export function useModelMetrics(id: number) {
  return useQuery({
    queryKey: modelKeys.metrics(id),
    queryFn: () => getModelMetrics(id),
    enabled: !!id,
  })
}

export function useActivateModel() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: activateModel,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: modelKeys.all })
    },
  })
}

export function useDeleteModel() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: deleteModel,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: modelKeys.all })
    },
  })
}

export function useInvalidateModels() {
  const queryClient = useQueryClient()
  
  return {
    invalidateAll: () => queryClient.invalidateQueries({ queryKey: modelKeys.all }),
    invalidateList: () => queryClient.invalidateQueries({ queryKey: modelKeys.lists() }),
    invalidateActive: () => queryClient.invalidateQueries({ queryKey: modelKeys.active() }),
    invalidateDetail: (id: number) => queryClient.invalidateQueries({ queryKey: modelKeys.detail(id) }),
  }
}
