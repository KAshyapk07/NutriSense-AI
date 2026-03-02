import type { ProcessResponse, SearchResponse, SearchFilters, ChatRequest, ChatResponseData, ChefParseRequest, ChefParseResponse } from './types'

const API_BASE = import.meta.env.VITE_API_URL ?? ''

export async function processQuery(
  query?: string,
  image?: File,
): Promise<ProcessResponse> {
  const form = new FormData()
  if (query) form.append('query', query)
  if (image) form.append('image', image)

  const res = await fetch(`${API_BASE}/process`, {
    method: 'POST',
    body: form,
  })

  if (!res.ok) {
    const text = await res.text()
    throw new Error(`Server error ${res.status}: ${text}`)
  }

  return res.json()
}

export async function searchQuery(
  q: string,
  filters: Partial<SearchFilters> = {},
): Promise<SearchResponse> {
  const params = new URLSearchParams()
  params.set('q', q)
  if (filters.cluster) params.set('cluster', filters.cluster)
  if (filters.limit) params.set('limit', String(filters.limit))
  if (filters.healthTags?.length) {
    for (const tag of filters.healthTags) params.append('health_tags', tag)
  }
  if (filters.excludeAllergens?.length) {
    for (const a of filters.excludeAllergens) params.append('exclude_allergens', a)
  }

  const res = await fetch(`${API_BASE}/search?${params.toString()}`)

  if (!res.ok) {
    const text = await res.text()
    throw new Error(`Server error ${res.status}: ${text}`)
  }

  return res.json()
}

export async function chatWithProduct(body: ChatRequest): Promise<ChatResponseData> {
  const res = await fetch(`${API_BASE}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })

  if (!res.ok) {
    const text = await res.text()
    throw new Error(`Server error ${res.status}: ${text}`)
  }

  return res.json()
}

export async function chefParse(body: ChefParseRequest): Promise<ChefParseResponse> {
  const res = await fetch(`${API_BASE}/chef/parse`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })

  if (!res.ok) {
    const text = await res.text()
    throw new Error(`Server error ${res.status}: ${text}`)
  }

  return res.json()
}
