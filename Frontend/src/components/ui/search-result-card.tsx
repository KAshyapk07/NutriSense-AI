import { motion, AnimatePresence } from 'framer-motion'
import { ChevronDown } from 'lucide-react'
import { useState } from 'react'
import { cn } from '@/lib/utils'
import type { SearchResult } from '@/lib/types'

/* ── Badge component (mirrors NutritionCard) ────────────────────── */
function Badge({
  children,
  variant = 'default',
}: {
  children: React.ReactNode
  variant?: 'default' | 'outline'
}) {
  return (
    <span
      className={cn(
        'inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium tracking-wide',
        variant === 'default'
          ? 'bg-[var(--color-accent)] text-[var(--color-accent-contrast)]'
          : 'border border-[var(--color-border)] text-[var(--color-text-muted)]',
      )}
    >
      {children}
    </span>
  )
}

/* ── Collapsible section (mirrors NutritionCard) ────────────────── */
function CollapsibleSection({
  title,
  children,
  defaultOpen = false,
}: {
  title: string
  children: React.ReactNode
  defaultOpen?: boolean
}) {
  const [open, setOpen] = useState(defaultOpen)

  return (
    <div className="border-t border-[var(--color-border)]">
      <button
        onClick={() => setOpen(!open)}
        className="flex w-full items-center justify-between px-6 py-4 text-left hover:bg-[var(--color-bg)] transition-colors"
      >
        <span className="text-base font-semibold uppercase tracking-widest text-[var(--color-text-muted)]">
          {title}
        </span>
        <motion.div
          animate={{ rotate: open ? 180 : 0 }}
          transition={{ duration: 0.2 }}
        >
          <ChevronDown size={16} className="text-[var(--color-text-muted)]" />
        </motion.div>
      </button>
      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.25, ease: 'easeInOut' }}
            className="overflow-hidden"
          >
            <div className="px-6 pb-5 text-base text-[var(--color-text)] leading-relaxed whitespace-pre-line">
              {children}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

/* ── Build ALL nutrition rows from a search result ──────────────── */
function buildNutritionRows(result: SearchResult): { label: string; value: string }[] {
  const rows: { label: string; value: string }[] = []

  if (result.cluster === 'recipe') {
    if (result.calories != null) rows.push({ label: 'Calories (kcal)', value: result.calories.toFixed(2) })
    if (result.carbohydrates != null) rows.push({ label: 'Carbohydrates (g)', value: result.carbohydrates.toFixed(2) })
    if (result.protein != null) rows.push({ label: 'Protein (g)', value: result.protein.toFixed(2) })
    if (result.fats != null) rows.push({ label: 'Fats (g)', value: result.fats.toFixed(2) })
    if (result.free_sugar != null) rows.push({ label: 'Free Sugar (g)', value: result.free_sugar.toFixed(2) })
    if (result.fibre != null) rows.push({ label: 'Fibre (g)', value: result.fibre.toFixed(2) })
    if (result.sodium != null) rows.push({ label: 'Sodium (mg)', value: result.sodium.toFixed(2) })
    if (result.calcium != null) rows.push({ label: 'Calcium (mg)', value: result.calcium.toFixed(2) })
    if (result.iron != null) rows.push({ label: 'Iron (mg)', value: result.iron.toFixed(2) })
    if (result.vitamin_c != null) rows.push({ label: 'Vitamin C (mg)', value: result.vitamin_c.toFixed(2) })
    if (result.folate != null) rows.push({ label: 'Folate (µg)', value: result.folate.toFixed(2) })
  } else {
    if (result.calories_100g != null) rows.push({ label: 'Calories / 100g (kcal)', value: result.calories_100g.toFixed(2) })
    if (result.carbohydrates_100g != null) rows.push({ label: 'Carbohydrates / 100g (g)', value: result.carbohydrates_100g.toFixed(2) })
    if (result.proteins_100g != null) rows.push({ label: 'Protein / 100g (g)', value: result.proteins_100g.toFixed(2) })
    if (result.fat_100g != null) rows.push({ label: 'Fat / 100g (g)', value: result.fat_100g.toFixed(2) })
    if (result.sugars_100g != null) rows.push({ label: 'Sugars / 100g (g)', value: result.sugars_100g.toFixed(2) })
    if (result.fiber_100g != null) rows.push({ label: 'Fibre / 100g (g)', value: result.fiber_100g.toFixed(2) })
    if (result.sodium_100g != null) rows.push({ label: 'Sodium / 100g (mg)', value: result.sodium_100g.toFixed(2) })
  }

  return rows
}

/* ── Main component ──────────────────────────────────────────────── */
interface SearchResultCardProps {
  result: SearchResult
  index?: number
  onChat?: (result: SearchResult) => void
}

export function SearchResultCard({
  result,
  index = 0,
  onChat,
}: SearchResultCardProps) {
  const isRecipe = result.cluster === 'recipe'
  const rows = buildNutritionRows(result)
  const scorePct = Math.round(result.final_score * 100)

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: index * 0.06, ease: 'easeOut' }}
      className="rounded-2xl border border-[var(--color-border)] bg-[var(--color-surface)] overflow-hidden"
    >
      {/* ── Header — matches NutritionCard header ── */}
      <div className="px-6 pt-6 pb-4">
        <h2 className="font-serif text-3xl font-bold text-[var(--color-text)] tracking-tight">
          {result.name}
        </h2>

        {/* Metadata row */}
        <div className="mt-1.5 flex flex-wrap items-center gap-2 text-sm text-[var(--color-text-muted)]">
          {isRecipe && result.cuisine && (
            <span className="rounded-full border border-[var(--color-border)] px-2.5 py-0.5 text-xs">
              {result.cuisine}
            </span>
          )}
          {isRecipe && result.prep_time_mins != null && (
            <span className="rounded-full border border-[var(--color-border)] px-2.5 py-0.5 text-xs">
              {result.prep_time_mins} min
            </span>
          )}
          {!isRecipe && result.brand && (
            <span className="rounded-full border border-[var(--color-border)] px-2.5 py-0.5 text-xs">
              {result.brand}
            </span>
          )}
          {!isRecipe && result.category && (
            <span className="rounded-full border border-[var(--color-border)] px-2.5 py-0.5 text-xs">
              {result.category}
            </span>
          )}
        </div>

        {/* Badges — confidence */}
        <div className="flex flex-wrap items-center gap-2 mt-3">
          <Badge>{scorePct}% Match</Badge>
        </div>
      </div>

      {/* ── Nutrition table — full rows, alternating colours ── */}
      {rows.length > 0 && (
        <div className="border-t border-[var(--color-border)]">
          {rows.map(({ label, value }, i) => (
            <div
              key={label}
              className={cn(
                'flex items-center justify-between px-6 py-3',
                i % 2 === 0
                  ? 'bg-[var(--color-surface)]'
                  : 'bg-[var(--color-bg)]',
              )}
            >
              <span className="text-base text-[var(--color-text-muted)]">{label}</span>
              <span className="text-base font-medium text-[var(--color-text)] tabular-nums">{value}</span>
            </div>
          ))}
        </div>
      )}

      {/* ── Ingredients (recipes only) ── */}
      {isRecipe && result.raw_ingredients && (
        <CollapsibleSection title="Ingredients" defaultOpen>
          {result.raw_ingredients}
        </CollapsibleSection>
      )}

      {/* ── Instructions (recipes only) ── */}
      {isRecipe && result.instructions && (
        <CollapsibleSection title="Instructions">
          {result.instructions}
        </CollapsibleSection>
      )}

      {/* ── Footer: Ask AI button ── */}
      <div className="border-t border-[var(--color-border)] px-6 py-3">
        <button
          onClick={(e) => { e.stopPropagation(); onChat?.(result) }}
          className="inline-flex items-center gap-2 rounded-lg border border-[var(--color-border)] text-sm font-medium text-[var(--color-text-muted)] px-4 py-2 hover:text-[var(--color-text)] hover:border-[var(--color-accent)]/50 transition-colors"
        >
          Ask AI about this
        </button>
      </div>
    </motion.div>
  )
}
