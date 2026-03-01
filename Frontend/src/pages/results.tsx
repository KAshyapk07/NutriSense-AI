/**
 * Chat page — ChatGPT-style AI conversation interface.
 * Route: /results (navigated from Home / Image / Compare / Modify pages)
 *
 * Features:
 * - Persistent conversation thread across multiple queries
 * - User queries appear as right-aligned accent bubbles
 * - AI responses render NutritionCard / ComparisonView / NutritionCard(modification)
 * - Extraction results show collapsible variant cards below the primary card
 * - Bottom input bar for follow-up queries (Enter to send, Shift+Enter for newline)
 */
import {
  useState,
  useRef,
  useEffect,
  useCallback,
  type FormEvent,
  type KeyboardEvent,
} from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import {
  AlertCircle,
  Send,
  RotateCcw,
  ChevronDown,
  ChevronUp,
  Sparkles,
} from 'lucide-react'
import { Header } from '@/components/layout/header'
import { NutritionCard } from '@/components/ui/nutrition-card'
import { ComparisonView } from '@/components/ui/comparison-view'
import { SearchResultCard } from '@/components/ui/search-result-card'
import { SkeletonLoader } from '@/components/ui/skeleton-loader'
import { processQuery } from '@/lib/api'
import {
  isComparison,
  isExtraction,
  isModification,
  isSearch,
  isError,
  type ProcessResponse,
  type ExtractionResponse,
  type RouterSearchResponse,
  type SearchResult,
} from '@/lib/types'
import { cn } from '@/lib/utils'

/* ── Chat message type ───────────────────────────────────────────── */

interface ChatMessage {
  id: string
  query: string
  imageFile?: File
  imagePreview?: string
  status: 'loading' | 'error' | 'done'
  result?: ProcessResponse
  error?: string
}

/* ── Variant card — shows as a full NutritionCard with toggle ──── */

function VariantCard({
  variant,
  index,
  defaultExpanded = false,
}: {
  variant: Record<string, unknown>
  index: number
  defaultExpanded?: boolean
}) {
  const [expanded, setExpanded] = useState(defaultExpanded)
  const name = (variant.recipe_name as string | undefined) ?? `Variant ${index + 1}`

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.08 }}
      className="rounded-2xl border border-[var(--color-border)] bg-[var(--color-surface)] overflow-hidden"
    >
      <button
        onClick={() => setExpanded((p) => !p)}
        className="w-full flex items-center justify-between px-6 py-4 text-left hover:bg-[var(--color-border)]/20 transition-colors"
      >
        <div>
          <p className="text-xs font-semibold text-[var(--color-accent)] uppercase tracking-wide mb-0.5">
            Top match {index + 2}
          </p>
          <p className="text-lg font-semibold text-[var(--color-text)]">{name}</p>
        </div>
        {expanded ? (
          <ChevronUp size={18} className="text-[var(--color-text-muted)]" />
        ) : (
          <ChevronDown size={18} className="text-[var(--color-text-muted)]" />
        )}
      </button>
      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.25 }}
            className="overflow-hidden"
          >
            <div className="px-4 pb-4">
              <NutritionCard
                dishName={name}
                nutrition={(variant.nutrition as ExtractionResponse['nutrition']) ?? null}
                ingredients={variant.ingredients as string | null | undefined}
                instructions={variant.instructions as string | null | undefined}
                llmResponse={variant.llm_response as string | null | undefined}
                confidence={variant.confidence as number | undefined}
                accuracy={variant.accuracy as number | undefined}
                source={variant.source as string | null | undefined}
                estimated={variant.estimated as boolean | undefined}
              />
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  )
}

/* ── Inline search results — rendered when Router returns pathway=search ── */

function SearchResultsInline({ data }: { data: RouterSearchResponse }) {
  const navigate = useNavigate()
  const recipeResults = data.results.filter((r) => r.cluster === 'recipe').slice(0, 3)
  const productResults = data.results.filter((r) => r.cluster === 'product').slice(0, 3)

  const handleOpenChat = (result: SearchResult) => {
    navigate('/product-chat', { state: { result } })
  }

  return (
    <div className="flex flex-col gap-6 w-full">
      {data.llm_response && (
        <p className="text-sm text-[var(--color-text-muted)] leading-relaxed px-1">
          {data.llm_response}
        </p>
      )}
      {recipeResults.length > 0 && (
        <div>
          <div className="flex items-center gap-2.5 mb-4">
            <h3 className="text-base font-bold text-[var(--color-text)]">Recipes ({recipeResults.length})</h3>
          </div>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
            {recipeResults.map((r, i) => (
              <SearchResultCard key={r.id} result={r} index={i} onChat={handleOpenChat} />
            ))}
          </div>
        </div>
      )}
      {productResults.length > 0 && (
        <div>
          <div className="flex items-center gap-2.5 mb-4">
            <h3 className="text-base font-bold text-[var(--color-text)]">Products ({productResults.length})</h3>
          </div>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
            {productResults.map((r, i) => (
              <SearchResultCard key={r.id} result={r} index={i} onChat={handleOpenChat} />
            ))}
          </div>
        </div>
      )}
      {recipeResults.length === 0 && productResults.length === 0 && (
        <p className="text-sm text-[var(--color-text-muted)] text-center py-8">
          No results found for this query.
        </p>
      )}
    </div>
  )
}

/* ── AI response area ────────────────────────────────────────────── */

function AiBubble({
  msg,
  onRetry,
}: {
  msg: ChatMessage
  onRetry: (msg: ChatMessage) => void
}) {
  if (msg.status === 'loading') {
    return (
      <div className="w-full max-w-4xl">
        <SkeletonLoader />
      </div>
    )
  }

  if (msg.status === 'error') {
    return (
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        className="max-w-4xl rounded-2xl border border-[var(--color-border)] bg-[var(--color-surface)] p-8 text-center"
      >
        <AlertCircle size={28} className="mx-auto text-[var(--color-text-muted)] mb-3 opacity-40" />
        <p className="text-base font-medium text-[var(--color-text)] mb-1">
          {msg.error?.includes('500') || msg.error?.includes('Failed to fetch')
            ? 'Service unavailable'
            : 'Something went wrong'}
        </p>
        <p className="text-sm text-[var(--color-text-muted)] mb-4 max-w-sm mx-auto leading-relaxed">
          {msg.error?.includes('Failed to fetch')
            ? 'The analysis service is unreachable. Make sure the backend is running on port 8000.'
            : msg.error}
        </p>
        <button
          onClick={() => onRetry(msg)}
          className="inline-flex items-center gap-2 rounded-full bg-[var(--color-accent)] text-[var(--color-accent-contrast)] px-4 py-2 text-xs font-medium hover:opacity-90 transition-opacity"
        >
          <RotateCcw size={12} />
          Retry
        </button>
      </motion.div>
    )
  }

  const { result } = msg
  if (!result) return null

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      className="flex flex-col gap-4 w-full max-w-screen-xl"
    >
      {/* Error from backend */}
      {isError(result) && (
        <div className="rounded-2xl border border-[var(--color-border)] bg-[var(--color-surface)] p-6 text-center">
          <AlertCircle size={28} className="mx-auto text-[var(--color-text-muted)] mb-3 opacity-40" />
          <p className="text-sm font-medium text-[var(--color-text)]">{result.error}</p>
          {result.detail && (
            <p className="text-xs text-[var(--color-text-muted)] mt-1">{result.detail}</p>
          )}
        </div>
      )}

      {/* Extraction */}
      {isExtraction(result) && (
        <>
          <NutritionCard
            dishName={result.recipe_name}
            nutrition={result.nutrition}
            ingredients={result.ingredients}
            instructions={result.instructions}
            llmResponse={result.llm_response}
            confidence={result.confidence}
            accuracy={result.accuracy}
            source={result.source}
            estimated={result.estimated}
          />
          {result.variants && result.variants.length > 0 && (
            <div className="flex flex-col gap-4">
              <p className="text-sm font-semibold text-[var(--color-text-muted)] uppercase tracking-wide px-1">
                Top {result.variants.length + 1} Results &mdash;{' '}
                {result.variants.length} more{' '}
                {result.variants.length === 1 ? 'match' : 'matches'}
              </p>
              {result.variants.map((v, i) => (
                <VariantCard key={i} variant={v} index={i} defaultExpanded />
              ))}
            </div>
          )}
        </>
      )}

      {/* Comparison */}
      {isComparison(result) && <ComparisonView data={result} />}

      {/* Modification */}
      {isModification(result) && (
        <NutritionCard
          dishName={result.recipe_name}
          nutrition={result.nutrition}
          ingredients={result.ingredients}
          instructions={result.instructions}
          llmResponse={result.llm_response}
          accuracy={result.accuracy}
          source={result.source}
          estimated={result.estimated}
          constraint={result.constraint}
        />
      )}

      {/* Search results — rendered inline as cards */}
      {isSearch(result) && (
        <SearchResultsInline data={result} />
      )}
    </motion.div>
  )
}

/* ═══════════════════════════════════════════════════════════════════
   Main chat page
═══════════════════════════════════════════════════════════════════ */

export default function Results() {
  const location = useLocation()
  const navigate = useNavigate()
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [inputValue, setInputValue] = useState('')
  const bottomRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const initialQuerySent = useRef(false)

  /* ── Send a query ── */
  const sendQuery = useCallback(async (query: string, imageFile?: File) => {
    const id = `${Date.now()}-${Math.random()}`
    const imagePreview = imageFile ? URL.createObjectURL(imageFile) : undefined

    setMessages((prev) => [
      ...prev,
      { id, query, imageFile, imagePreview, status: 'loading' },
    ])

    try {
      const result = await processQuery(query || undefined, imageFile)
      setMessages((prev) =>
        prev.map((m) => (m.id === id ? { ...m, status: 'done', result } : m)),
      )
    } catch (err) {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === id
            ? {
                ...m,
                status: 'error',
                error: err instanceof Error ? err.message : 'An unexpected error occurred',
              }
            : m,
        ),
      )
    }
  }, [])

  /* ── Retry failed message ── */
  const retryMessage = useCallback(
    (msg: ChatMessage) => {
      setMessages((prev) => prev.filter((m) => m.id !== msg.id))
      sendQuery(msg.query, msg.imageFile)
    },
    [sendQuery],
  )

  /* ── Seed from navigation state on mount ── */
  useEffect(() => {
    if (initialQuerySent.current) return   // guard against React StrictMode double-invoke
    const state = location.state as { query?: string; image?: File } | null
    if (state?.query || state?.image) {
      initialQuerySent.current = true
      sendQuery(state.query ?? '', state.image)
      // Clear state so refresh doesn't re-trigger
      window.history.replaceState({}, '')
    } else if (messages.length === 0) {
      navigate('/')
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  /* ── Scroll to bottom ── */
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  /* ── Auto-grow textarea ── */
  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInputValue(e.target.value)
    const el = e.target
    el.style.height = 'auto'
    el.style.height = `${el.scrollHeight}px`
  }

  const handleSubmit = (e?: FormEvent) => {
    e?.preventDefault()
    const q = inputValue.trim()
    if (!q) return
    setInputValue('')
    if (textareaRef.current) textareaRef.current.style.height = 'auto'
    sendQuery(q)
  }

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  const isAnyLoading = messages.some((m) => m.status === 'loading')

  return (
    <div className="flex flex-col min-h-screen bg-[var(--color-bg)]">
      <Header />

      {/* ── Message thread ── */}
      <main className="flex-1 mx-auto w-full max-w-screen-2xl px-6 sm:px-8 lg:px-16 pt-8 pb-36">
        <div className="flex flex-col gap-10">
          {messages.map((msg) => (
            <div key={msg.id} className="flex flex-col gap-4">
              {/* ── User bubble (right-aligned) ── */}
              <div className="flex justify-end">
                <div className="flex flex-col items-end gap-2 max-w-[75%]">
                  {msg.imagePreview && (
                    <div className="h-20 w-20 rounded-xl overflow-hidden border border-[var(--color-border)]">
                      <img
                        src={msg.imagePreview}
                        alt="Uploaded food"
                        className="h-full w-full object-cover"
                      />
                    </div>
                  )}
                  {msg.query && (
                    <div className="rounded-2xl rounded-tr-sm bg-[var(--color-accent)] text-[var(--color-accent-contrast)] px-5 py-3 text-base font-medium">
                      {msg.query}
                    </div>
                  )}
                </div>
              </div>

              {/* ── AI response (left-aligned) ── */}
              <div className="flex items-start gap-3">
                <div className="mt-0.5 shrink-0 flex h-8 w-8 items-center justify-center rounded-full bg-[var(--color-accent)]/10 border border-[var(--color-accent)]/20">
                  <Sparkles size={14} className="text-[var(--color-accent)]" />
                </div>
                <AiBubble msg={msg} onRetry={retryMessage} />
              </div>
            </div>
          ))}
        </div>
        <div ref={bottomRef} />
      </main>

      {/* ── Fixed bottom input bar ── */}
      <div className="fixed bottom-0 left-0 right-0 z-30 border-t border-[var(--color-border)] bg-[var(--color-bg)]/95 backdrop-blur-md">
        <form
          onSubmit={handleSubmit}
          className="mx-auto max-w-screen-2xl px-6 sm:px-8 lg:px-16 py-3"
        >
          <div
            className={cn(
              'flex items-end gap-3 rounded-2xl border px-4 py-3',
              'border-[var(--color-border)] bg-[var(--color-surface)]',
              'focus-within:border-[var(--color-accent)]/50 transition-colors duration-150',
            )}
          >
            <textarea
              ref={textareaRef}
              rows={1}
              value={inputValue}
              onChange={handleInputChange}
              onKeyDown={handleKeyDown}
              placeholder="Ask a follow-up... e.g. 'compare with chicken biryani', 'make it low-calorie'"
              className={cn(
                'flex-1 resize-none bg-transparent text-base text-[var(--color-text)]',
                'placeholder-[var(--color-text-muted)] outline-none leading-relaxed max-h-36',
              )}
            />
            <button
              type="submit"
              disabled={!inputValue.trim() || isAnyLoading}
              className={cn(
                'shrink-0 flex h-8 w-8 items-center justify-center rounded-full transition-all duration-150',
                inputValue.trim() && !isAnyLoading
                  ? 'bg-[var(--color-accent)] text-[var(--color-accent-contrast)] hover:opacity-90'
                  : 'bg-[var(--color-border)] text-[var(--color-text-muted)] cursor-not-allowed',
              )}
            >
              <Send size={14} />
            </button>
          </div>
          <p className="mt-1.5 text-center text-xs text-[var(--color-text-muted)]">
            Enter to send · Shift+Enter for new line
          </p>
        </form>
      </div>
    </div>
  )
}
