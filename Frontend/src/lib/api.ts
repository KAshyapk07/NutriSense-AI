import type { ProcessResponse } from './types'

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
