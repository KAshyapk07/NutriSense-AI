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

export interface RouterSearchResponse {
  pathway: 'search'
  query: string
  results: SearchResult[]
  total: number
  llm_response?: string | null
}

export type ProcessResponse =
  | ExtractionResponse
  | ComparisonResponse
  | ModificationResponse
  | RouterSearchResponse
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

export function isSearch(r: ProcessResponse): r is RouterSearchResponse {
  return 'pathway' in r && r.pathway === 'search'
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

/* ──────────────────────────────────────────────
   Phase 5 — AI Chef Agent types
   Mirroring Backend/schemas/chef.py
   ────────────────────────────────────────────── */

export interface MiseEnPlaceItem {
  id: number
  text: string
  duration_minutes: number | null
}

export interface CookStep {
  id: number
  action: string
  timer_seconds: number | null
  tool: string | null
  tip: string | null
}

export interface ChefParseRequest {
  recipe_name: string
  instructions?: string | null
  ingredients?: string | null
}

export interface ChefParseResponse {
  recipe_name: string
  mise_en_place: MiseEnPlaceItem[]
  steps: CookStep[]
  tools_required: string[]
  estimated_total_minutes: number | null
  parse_error: string | null
}

/* ──────────────────────────────────────────────
   Phase 5.5 — P2P Voice Kitchen Remote types
   ────────────────────────────────────────────── */

export type VoiceAction =
  | 'NEXT'
  | 'PREV'
  | 'DONE'
  | 'STRIKE'
  | 'TIMER_START'
  | 'TIMER_PAUSE'
  | 'TIMER_RESET'
  | 'REPEAT'
  | 'ASK'
  | 'NOOP'

export interface ChefIntentRequest {
  raw_text: string
  recipe_name: string
  current_step: number
  total_steps: number
  current_action: string
  timer_running: boolean
  timer_seconds_left: number | null
}

export interface ChefIntentResponse {
  action: VoiceAction
  step: number | null
  question: string | null
  confidence: number
  filtered: boolean
}

export interface CookingSessionState {
  recipe_name: string
  current_step: number
  total_steps: number
  current_action: string
  current_tool: string | null
  current_tip: string | null
  timer_total: number | null
  timer_left: number | null
  timer_running: boolean
  completed_steps: number[]
  phase: 'prep' | 'cooking' | 'done'
  steps_overview: Array<{ id: string; action: string; completed: string }>
}
