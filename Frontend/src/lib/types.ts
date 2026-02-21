/* ──────────────────────────────────────────────
   TypeScript types mirroring Backend/schemas/process.py
   ────────────────────────────────────────────── */

export interface NutritionData {
  [key: string]: string | number
}

export interface ExtractionResponse {
  pathway: 'extraction'
  status?: string | null
  recipe_name?: string | null
  confidence?: number
  nutrition?: NutritionData | null
  ingredients?: string | null
  instructions?: string | null
  meta?: Record<string, unknown>
  variants?: Record<string, unknown>[]
  llm_response?: string | null
  accuracy?: number
  source?: string | null
  estimated?: boolean
}

export interface ComparisonResponse {
  pathway: 'comparison'
  dish_a?: string | null
  nutrition_a?: NutritionData | null
  dish_b?: string | null
  nutrition_b?: NutritionData | null
  llm_response?: string | null
  goal?: string | null
  estimated?: boolean
  accuracy?: number
  source?: string | null
}

export interface ModificationResponse {
  pathway: 'modification'
  recipe_name?: string | null
  constraint?: string | null
  nutrition?: NutritionData | null
  ingredients?: string | null
  instructions?: string | null
  llm_response?: string | null
  accuracy?: number
  source?: string | null
  estimated?: boolean
}

export interface ErrorResponse {
  error: string
  detail?: string | null
}

export type ProcessResponse =
  | ExtractionResponse
  | ComparisonResponse
  | ModificationResponse
  | ErrorResponse

/* ── Type guards ── */

export function isExtraction(r: ProcessResponse): r is ExtractionResponse {
  return 'pathway' in r && r.pathway === 'extraction'
}

export function isComparison(r: ProcessResponse): r is ComparisonResponse {
  return 'pathway' in r && r.pathway === 'comparison'
}

export function isModification(r: ProcessResponse): r is ModificationResponse {
  return 'pathway' in r && r.pathway === 'modification'
}

export function isError(r: ProcessResponse): r is ErrorResponse {
  return 'error' in r
}
