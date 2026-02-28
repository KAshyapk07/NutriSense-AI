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

/* ──────────────────────────────────────────────
   Phase 3 — GraphRAG Semantic Search types
   Mirroring Backend/schemas/search.py
   ────────────────────────────────────────────── */

export interface SearchResult {
  id: string
  name: string
  cluster: 'recipe' | 'product'
  vector_score: number
  graph_score: number
  final_score: number
  // Recipe fields
  food_name?: string | null
  cuisine?: string | null
  prep_time_mins?: number | null
  calories?: number | null
  protein?: number | null
  carbohydrates?: number | null
  fats?: number | null
  fibre?: number | null
  free_sugar?: number | null
  sodium?: number | null
  calcium?: number | null
  iron?: number | null
  vitamin_c?: number | null
  folate?: number | null
  raw_ingredients?: string | null
  instructions?: string | null
  // Product fields
  brand?: string | null
  category?: string | null
  nutriscore_grade?: string | null
  nova_group?: number | null
  calories_100g?: number | null
  proteins_100g?: number | null
  carbohydrates_100g?: number | null
  fat_100g?: number | null
  fiber_100g?: number | null
  sodium_100g?: number | null
  sugars_100g?: number | null
  image_url?: string | null
}

export interface SearchResponse {
  query: string
  cluster_filter: 'all' | 'recipe' | 'product'
  health_tags: string[]
  excluded_allergens: string[]
  total: number
  results: SearchResult[]
  vector_search_used: boolean
  health_tags_available: string[]
}

export interface SearchFilters {
  cluster: 'all' | 'recipe' | 'product'
  healthTags: string[]
  excludeAllergens: string[]
  limit: number
}

/* ──────────────────────────────────────────────
   Phase 4 — Product Chat types
   ────────────────────────────────────────────── */

export interface ChatMessagePayload {
  role: 'user' | 'assistant'
  content: string
}

export interface ChatRequest {
  message: string
  context?: Record<string, unknown> | null
  history?: ChatMessagePayload[]
}

export interface ChatResponseData {
  reply: string
}
