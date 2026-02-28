/**
 * Search page  GraphRAG semantic search with filters.
 *
 * Route: /search?q=<query>
 * Features:
 * - Table-style nutrition cards grouped by Recipes and Products
 * - 3 recipes + 3 products max per search
 * - Filter panel (cluster toggle + allergen pills)
 * - "Ask AI" navigates to a dedicated chat page
 */
import { useEffect, useState, useCallback } from 'react'
import { useSearchParams, useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import {
  AlertCircle,
  Network,
  Search,
  SlidersHorizontal,
  Zap,
  SearchX,
  RotateCcw,
  UtensilsCrossed,
  Package,
} from 'lucide-react'
import { Header } from '@/components/layout/header'
import { FilterPanel } from '@/components/ui/filter-panel'
import { SearchResultCard } from '@/components/ui/search-result-card'
import { SkeletonLoader } from '@/components/ui/skeleton-loader'
import { useSemanticSearch } from '@/hooks/use-semantic-search'
import { usePreferences } from '@/hooks/use-preferences'
import { cn } from '@/lib/utils'
import type { SearchFilters, SearchResult } from '@/lib/types'

const MAX_PER_CLUSTER = 3

/* -- Empty state -------------------------------------------------- */
function EmptyState({ query }: { query: string }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      className="flex flex-col items-center justify-center py-24 text-center"
    >
      <SearchX size={48} className="text-[var(--color-text-muted)] mb-5 opacity-40" />
      <p className="text-base font-medium text-[var(--color-text)] mb-2">
        No results found for &ldquo;{query}&rdquo;
      </p>
      <p className="text-sm text-[var(--color-text-muted)] max-w-sm leading-relaxed">
        Try a different spelling or fewer filters.
      </p>
    </motion.div>
  )
}

/* -- Section heading ---------------------------------------------- */
function SectionHeading({
  icon: Icon,
  label,
  count,
}: {
  icon: React.FC<{ size?: number; className?: string }>
  label: string
  count: number
}) {
  return (
    <div className="flex items-center gap-3 mb-5 mt-10 first:mt-0">
      <Icon size={22} className="text-[var(--color-accent)]" />
      <h2 className="text-lg font-bold text-[var(--color-text)]">{label}</h2>
      <span className="text-sm text-[var(--color-text-muted)]">({count})</span>
    </div>
  )
}

/* =================================================================
   Main search page
   ================================================================= */
export default function SearchPage() {
  const navigate = useNavigate()
  const [searchParams, setSearchParams] = useSearchParams()
  const { prefs } = usePreferences()

  const q = searchParams.get('q') ?? ''

  const { loading, response, error, search, reset } = useSemanticSearch()
  const [filters, setFilters] = useState<SearchFilters>(() => ({
    cluster: 'all',
    healthTags: prefs.activeDiets,
    excludeAllergens: prefs.excludeAllergens,
    limit: 10,
  }))
  const [showFilters, setShowFilters] = useState(false)

  /* -- Run search whenever query or filters change -- */
  const runSearch = useCallback(
    (query: string, f: SearchFilters) => {
      if (!query.trim()) return
      search(query, f)
    },
    [search],
  )

  useEffect(() => {
    if (q) runSearch(q, filters)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [q])

  useEffect(() => {
    if (q) runSearch(q, filters)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [filters])

  const handleNewSearch = (query: string) => {
    reset()
    setSearchParams({ q: query })
  }

  const handleFilterChange = (f: SearchFilters) => {
    setFilters(f)
  }

  const handleOpenChat = (result: SearchResult) => {
    navigate('/product-chat', { state: { result } })
  }

  /* -- No query -> empty prompt -- */
  if (!q) {
    return (
      <div className="min-h-screen bg-[var(--color-bg)]">
        <Header />
        <div className="flex flex-col items-center justify-center py-32 px-4 text-center">
          <Search size={48} className="text-[var(--color-text-muted)] mb-5 opacity-40" />
          <p className="text-lg font-medium text-[var(--color-text)] mb-2">Search for food</p>
          <p className="text-sm text-[var(--color-text-muted)] max-w-md leading-relaxed">
            Use the search bar above to find recipes and products across our database
            of 725+ Indian recipes and 6,400+ packaged products.
          </p>
        </div>
      </div>
    )
  }

  const results = response?.results ?? []
  const recipeResults = results.filter((r) => r.cluster === 'recipe').slice(0, MAX_PER_CLUSTER)
  const productResults = results.filter((r) => r.cluster === 'product').slice(0, MAX_PER_CLUSTER)
  const displayedTotal = recipeResults.length + productResults.length

  return (
    <div className="min-h-screen bg-[var(--color-bg)]">
      <Header onSearch={handleNewSearch} defaultQuery={q} />

      <div className="mx-auto max-w-screen-2xl px-6 sm:px-8 lg:px-16 py-8">
        {/* -- Page header -- */}
        <div className="flex items-center justify-between mb-6 gap-4">
          <div>
            <h1 className="text-xl font-bold text-[var(--color-text)]">
              Results for{' '}
              <span className="text-[var(--color-text-muted)] font-normal">&ldquo;{q}&rdquo;</span>
            </h1>
            {response && !loading && (
              <div className="mt-2 flex flex-wrap items-center gap-3">
                <span className="text-sm text-[var(--color-text-muted)]">
                  Showing {displayedTotal} of {response.total} result{response.total !== 1 ? 's' : ''}
                  {filters.cluster === 'all' && displayedTotal > 0 && (
                    <> &mdash; {recipeResults.length} recipes, {productResults.length} products</>
                  )}
                </span>
                {response.vector_search_used && (
                  <span className="inline-flex items-center gap-1.5 rounded-full border border-emerald-500/30 bg-emerald-500/10 px-3 py-1 text-xs font-semibold text-emerald-500">
                    <Zap size={11} />
                    GraphRAG
                  </span>
                )}
                {!response.vector_search_used && (
                  <span className="inline-flex items-center gap-1.5 rounded-full border border-[var(--color-border)] px-3 py-1 text-xs text-[var(--color-text-muted)]">
                    <Network size={11} />
                    Full-text
                  </span>
                )}
              </div>
            )}
          </div>

          {/* Filter toggle */}
          <button
            onClick={() => setShowFilters((v) => !v)}
            className={cn(
              'flex items-center gap-2 rounded-xl border px-4 py-2.5 text-sm font-medium transition-colors duration-150',
              showFilters
                ? 'border-[var(--color-accent)] bg-[var(--color-accent)] text-[var(--color-accent-contrast)]'
                : 'border-[var(--color-border)] text-[var(--color-text-muted)] hover:text-[var(--color-text)]',
            )}
          >
            <SlidersHorizontal size={15} />
            Filters
          </button>
        </div>

        {/* -- Collapsible filter bar -- */}
        <AnimatePresence>
          {showFilters && (
            <motion.div
              key="filter-bar"
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: 0.2 }}
              className="overflow-hidden mb-6"
            >
              <FilterPanel
                filters={filters}
                availableHealthTags={response?.health_tags_available ?? []}
                onChange={handleFilterChange}
                onClose={() => setShowFilters(false)}
              />
            </motion.div>
          )}
        </AnimatePresence>

        {/* -- Loading -- */}
        {loading && <SkeletonLoader />}

        {/* -- Error -- */}
        {error && !loading && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="rounded-2xl border border-[var(--color-border)] bg-[var(--color-surface)] p-10 text-center"
          >
            <AlertCircle size={40} className="mx-auto text-[var(--color-text-muted)] mb-4 opacity-40" />
            <p className="text-base font-medium text-[var(--color-text)] mb-2">Search failed</p>
            <p className="text-sm text-[var(--color-text-muted)] mb-6 max-w-sm mx-auto">{error}</p>
            <button
              onClick={() => runSearch(q, filters)}
              className="inline-flex items-center gap-2 rounded-full border border-[var(--color-border)] px-5 py-2 text-sm font-medium text-[var(--color-text-muted)] hover:text-[var(--color-text)] transition-colors"
            >
              <RotateCcw size={14} />
              Retry
            </button>
          </motion.div>
        )}

        {/* -- Empty state -- */}
        {!loading && !error && response && response.total === 0 && (
          <EmptyState query={q} />
        )}

        {/* -- Results -- */}
        {!loading && !error && (recipeResults.length > 0 || productResults.length > 0) && (
          <>
            {/* Recipes */}
            {recipeResults.length > 0 && (
              <div>
                <SectionHeading icon={UtensilsCrossed} label="Recipes" count={recipeResults.length} />
                <div className="grid grid-cols-1 lg:grid-cols-2 2xl:grid-cols-3 gap-6">
                  {recipeResults.map((result, i) => (
                    <SearchResultCard
                      key={result.id}
                      result={result}
                      index={i}
                      onChat={handleOpenChat}
                    />
                  ))}
                </div>
              </div>
            )}

            {/* Products */}
            {productResults.length > 0 && (
              <div>
                <SectionHeading icon={Package} label="Products" count={productResults.length} />
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  {productResults.map((result, i) => (
                    <SearchResultCard
                      key={result.id}
                      result={result}
                      index={i}
                      onChat={handleOpenChat}
                    />
                  ))}
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}
