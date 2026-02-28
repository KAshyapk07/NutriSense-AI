/**
 * Modify Recipe page — adapt any recipe to dietary needs.
 * Route: /modify
 *
 * User enters a recipe name + a dietary constraint, submits to AI.
 * Navigates to /results (chat interface) with the combined query.
 */
import { useState, type FormEvent } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { Sparkles } from 'lucide-react'
import { Header } from '@/components/layout/header'
import { cn } from '@/lib/utils'

const PRESET_CONSTRAINTS = [
  'low-calorie',
  'gluten-free',
  'vegan',
  'high-protein',
  'low-carb',
  'no dairy',
  'diabetic-friendly',
  'low-sodium',
]

export default function ModifyPage() {
  const navigate = useNavigate()
  const [recipe, setRecipe] = useState('')
  const [constraint, setConstraint] = useState('')

  const canSubmit = recipe.trim().length > 1

  const handleSubmit = (e?: FormEvent) => {
    e?.preventDefault()
    if (!canSubmit) return
    const q = constraint.trim()
      ? `Modify ${recipe.trim()} to be ${constraint.trim()}`
      : `Suggest a healthier version of ${recipe.trim()}`
    navigate('/results', { state: { query: q } })
  }

  const applyPreset = (preset: string) => {
    setConstraint((prev) =>
      prev.includes(preset) ? prev.replace(preset, '').trim() : `${prev} ${preset}`.trim(),
    )
  }

  return (
    <div className="min-h-screen bg-[var(--color-bg)]">
      <Header />
      <main className="mx-auto max-w-4xl px-4 sm:px-6 lg:px-12 py-14">
        {/* Heading */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-10 text-center"
        >
          <p className="text-xs font-semibold uppercase tracking-widest text-[var(--color-accent)] mb-2">
            Modify Recipe
          </p>
          <h1 className="text-3xl font-bold text-[var(--color-text)] mb-3">
            Adapt to your needs
          </h1>
          <p className="text-sm text-[var(--color-text-muted)] max-w-sm mx-auto leading-relaxed">
            Enter any dish and describe how to modify it — the AI rewrites
            ingredients, instructions, and recalculates nutrition.
          </p>
        </motion.div>

        {/* Form */}
        <motion.form
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.12 }}
          onSubmit={handleSubmit}
          className="flex flex-col gap-5"
        >
          {/* Recipe input */}
          <div>
            <label className="mb-1.5 block text-xs font-semibold uppercase tracking-wide text-[var(--color-text-muted)]">
              Recipe
            </label>
            <input
              type="text"
              value={recipe}
              onChange={(e) => setRecipe(e.target.value)}
              placeholder="e.g. Butter Chicken, Samosa, Biryani…"
              className={cn(
                'w-full rounded-2xl border border-[var(--color-border)] bg-[var(--color-surface)]',
                'px-4 py-3 text-sm text-[var(--color-text)] placeholder-[var(--color-text-muted)]',
                'outline-none focus:border-[var(--color-accent)]/50 transition-colors',
              )}
            />
          </div>

          {/* Constraint input */}
          <div>
            <label className="mb-1.5 block text-xs font-semibold uppercase tracking-wide text-[var(--color-text-muted)]">
              Dietary constraint <span className="normal-case font-normal">(optional)</span>
            </label>
            <input
              type="text"
              value={constraint}
              onChange={(e) => setConstraint(e.target.value)}
              placeholder="e.g. vegan, low-calorie, gluten-free…"
              className={cn(
                'w-full rounded-2xl border border-[var(--color-border)] bg-[var(--color-surface)]',
                'px-4 py-3 text-sm text-[var(--color-text)] placeholder-[var(--color-text-muted)]',
                'outline-none focus:border-[var(--color-accent)]/50 transition-colors',
              )}
            />
          </div>

          {/* Preset chips */}
          <div>
            <p className="mb-2 text-xs text-[var(--color-text-muted)]">Quick presets:</p>
            <div className="flex flex-wrap gap-2">
              {PRESET_CONSTRAINTS.map((p) => {
                const active = constraint.includes(p)
                return (
                  <button
                    key={p}
                    type="button"
                    onClick={() => applyPreset(p)}
                    className={cn(
                      'rounded-full border px-3 py-1 text-xs font-medium transition-all',
                      active
                        ? 'border-[var(--color-accent)] bg-[var(--color-accent)] text-[var(--color-accent-contrast)]'
                        : 'border-[var(--color-border)] text-[var(--color-text-muted)] hover:border-[var(--color-accent)]/50 hover:text-[var(--color-text)]',
                    )}
                  >
                    {p}
                  </button>
                )
              })}
            </div>
          </div>

          {/* Submit */}
          <button
            type="submit"
            disabled={!canSubmit}
            className={cn(
              'mt-2 inline-flex w-full items-center justify-center gap-2 rounded-full py-3 text-sm font-semibold transition-all',
              canSubmit
                ? 'bg-[var(--color-accent)] text-[var(--color-accent-contrast)] hover:opacity-90 shadow-lg shadow-[var(--color-accent)]/20'
                : 'bg-[var(--color-border)] text-[var(--color-text-muted)] cursor-not-allowed',
            )}
          >
            <Sparkles size={16} />
            Modify with AI
          </button>
        </motion.form>
      </main>
    </div>
  )
}
