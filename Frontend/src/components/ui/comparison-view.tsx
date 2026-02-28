import { motion } from 'framer-motion'
import { NutritionCard } from './nutrition-card'
import type { ComparisonResponse } from '@/lib/types'
import { cn } from '@/lib/utils'

interface ComparisonViewProps {
  data: ComparisonResponse
  className?: string
}

export function ComparisonView({ data, className }: ComparisonViewProps) {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.4 }}
      className={cn('w-full', className)}
    >
      {/* VS Header */}
      <div className="flex items-center justify-center gap-4 mb-8">
        <span className="font-serif text-xl font-bold text-[var(--color-text)] tracking-tight">
          {data.dish_a ?? 'Dish A'}
        </span>
        <span className="flex items-center justify-center w-10 h-10 rounded-full border border-[var(--color-border)] text-xs font-bold uppercase tracking-widest text-[var(--color-text-muted)]">
          vs
        </span>
        <span className="font-serif text-xl font-bold text-[var(--color-text)] tracking-tight">
          {data.dish_b ?? 'Dish B'}
        </span>
      </div>

      {/* Side-by-side cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <NutritionCard
          dishName={data.dish_a}
          nutrition={data.nutrition_a}
          estimated={data.estimated}
          accuracy={data.accuracy}
          source={data.source}
        />
        <NutritionCard
          dishName={data.dish_b}
          nutrition={data.nutrition_b}
          estimated={data.estimated}
          accuracy={data.accuracy}
          source={data.source}
        />
      </div>

      {/* LLM Analysis */}
      {data.llm_response && (
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3, duration: 0.4 }}
          className="mt-8 rounded-2xl border border-[var(--color-border)] bg-[var(--color-surface)] p-6"
        >
          <h3 className="text-base font-semibold uppercase tracking-widest text-[var(--color-text-muted)] mb-4">
            Comparison Analysis
          </h3>
          <div className="text-base leading-relaxed text-[var(--color-text)] whitespace-pre-line">
            {data.llm_response}
          </div>
        </motion.div>
      )}
    </motion.div>
  )
}
