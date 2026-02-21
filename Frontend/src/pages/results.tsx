import { useEffect } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { AlertCircle, RotateCcw } from 'lucide-react'
import { Header } from '@/components/layout/header'
import { NutritionCard } from '@/components/ui/nutrition-card'
import { ComparisonView } from '@/components/ui/comparison-view'
import { SkeletonLoader } from '@/components/ui/skeleton-loader'
import { useSearch } from '@/hooks/use-search'
import {
  isComparison,
  isExtraction,
  isModification,
  isError,
} from '@/lib/types'

export default function Results() {
  const location = useLocation()
  const navigate = useNavigate()
  const { loading, result, error, imagePreview, search } = useSearch()

  const stateQuery = (location.state as { query?: string })?.query ?? ''
  const stateImage = (location.state as { image?: File })?.image

  // Trigger search on mount
  useEffect(() => {
    if (stateQuery || stateImage) {
      search(stateQuery, stateImage)
    } else {
      navigate('/')
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const handleNewSearch = (query: string, image?: File) => {
    search(query, image)
    // Update URL state so back-button works
    window.history.replaceState({ query, image }, '')
  }

  return (
    <div className="min-h-screen bg-[var(--color-bg)]">
      <Header onSearch={handleNewSearch} defaultQuery={stateQuery} />

      <main className="mx-auto max-w-4xl px-4 sm:px-6 py-10">
        {/* Image preview */}
        {imagePreview && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="mb-6 flex items-center gap-4"
          >
            <div className="h-16 w-16 rounded-xl overflow-hidden border border-[var(--color-border)]">
              <img
                src={imagePreview}
                alt="Uploaded food"
                className="h-full w-full object-cover"
              />
            </div>
            <div>
              <p className="text-sm font-medium text-[var(--color-text)]">
                Image uploaded
              </p>
              <p className="text-xs text-[var(--color-text-muted)]">
                Analyzing the dish...
              </p>
            </div>
          </motion.div>
        )}

        {/* Loading */}
        {loading && <SkeletonLoader />}

        {/* Error */}
        {error && (
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            className="rounded-2xl border border-[var(--color-border)] bg-[var(--color-surface)] p-8 text-center"
          >
            <AlertCircle
              size={32}
              className="mx-auto text-[var(--color-text-muted)] mb-4"
            />
            <p className="text-sm font-medium text-[var(--color-text)] mb-2">
              Something went wrong
            </p>
            <p className="text-xs text-[var(--color-text-muted)] mb-6 max-w-md mx-auto">
              {error}
            </p>
            <button
              onClick={() => search(stateQuery, stateImage)}
              className="inline-flex items-center gap-2 rounded-full bg-[var(--color-accent)] text-[var(--color-accent-contrast)] px-5 py-2 text-sm font-medium hover:opacity-90 transition-opacity"
            >
              <RotateCcw size={14} />
              Try again
            </button>
          </motion.div>
        )}

        {/* Results */}
        {result && !loading && (
          <>
            {/* Extraction pathway */}
            {isExtraction(result) && (
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
            )}

            {/* Comparison pathway */}
            {isComparison(result) && <ComparisonView data={result} />}

            {/* Modification pathway */}
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

            {/* Error response from backend */}
            {isError(result) && (
              <motion.div
                initial={{ opacity: 0, y: 16 }}
                animate={{ opacity: 1, y: 0 }}
                className="rounded-2xl border border-[var(--color-border)] bg-[var(--color-surface)] p-8 text-center"
              >
                <AlertCircle
                  size={32}
                  className="mx-auto text-[var(--color-text-muted)] mb-4"
                />
                <p className="text-sm font-medium text-[var(--color-text)] mb-1">
                  {result.error}
                </p>
                {result.detail && (
                  <p className="text-xs text-[var(--color-text-muted)]">
                    {result.detail}
                  </p>
                )}
              </motion.div>
            )}
          </>
        )}
      </main>
    </div>
  )
}
