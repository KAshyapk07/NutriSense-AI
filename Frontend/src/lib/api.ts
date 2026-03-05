import type { ProcessResponse, SearchResponse, SearchFilters, ChatRequest, ChatResponseData, ChefParseRequest, ChefParseResponse, ChefIntentRequest, ChefIntentResponse } from './types'

const API_BASE = import.meta.env.VITE_API_URL ?? ''

// Header that tells ngrok to skip its browser interstitial page.
// Without this, API fetch() calls through an ngrok tunnel receive an HTML
// warning page instead of the expected JSON response.
const COMMON_HEADERS: Record<string, string> = {
  'ngrok-skip-browser-warning': '1',
}

// ── App config (exposes PUBLIC_URL set in backend .env) ──────────────────────
export interface AppConfig {
  /** Publicly reachable base URL of the server (e.g. ngrok or deployed domain). */
  remote_base_url: string
  /** "production" when PUBLIC_URL is set, "local" otherwise. */
  deployment: 'production' | 'local'
}

/**
 * Fetch runtime config from the backend.  Used by the P2P Kitchen Remote
 * feature to build the correct QR code URL when the app is behind ngrok or
 * deployed to a public host.
 *
 * Falls back gracefully — never throws; returns empty strings on error.
 */
export async function getAppConfig(): Promise<AppConfig> {
  try {
    const res = await fetch(`${API_BASE}/config`, { headers: COMMON_HEADERS })
    if (!res.ok) return { remote_base_url: '', deployment: 'local' }
    return res.json()
  } catch {
    return { remote_base_url: '', deployment: 'local' }
  }
}

export async function processQuery(
  query?: string,
  image?: File,
): Promise<ProcessResponse> {
  const form = new FormData()
  if (query) form.append('query', query)
  if (image) form.append('image', image)

  const res = await fetch(`${API_BASE}/process`, {
    method: 'POST',
    headers: COMMON_HEADERS,
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

  const res = await fetch(`${API_BASE}/search?${params.toString()}`, {
    headers: COMMON_HEADERS,
  })

  if (!res.ok) {
    const text = await res.text()
    throw new Error(`Server error ${res.status}: ${text}`)
  }

  return res.json()
}

export async function chatWithProduct(body: ChatRequest): Promise<ChatResponseData> {
  const res = await fetch(`${API_BASE}/chat`, {
    method: 'POST',
    headers: { ...COMMON_HEADERS, 'Content-Type': 'application/json' },
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
    headers: { ...COMMON_HEADERS, 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })

  if (!res.ok) {
    const text = await res.text()
    throw new Error(`Server error ${res.status}: ${text}`)
  }

  return res.json()
}

export async function chefIntent(body: ChefIntentRequest): Promise<ChefIntentResponse> {
  const res = await fetch(`${API_BASE}/chef/intent`, {
    method: 'POST',
    headers: { ...COMMON_HEADERS, 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })

  if (!res.ok) {
    const text = await res.text()
    throw new Error(`Server error ${res.status}: ${text}`)
  }

  return res.json()
}
