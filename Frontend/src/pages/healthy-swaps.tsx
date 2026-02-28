/**
 * Healthy Swaps page — lighter alternatives for your favourite dishes.
 * Route: /healthy-swaps
 *
 * User enters a dish; AI suggests healthier alternatives with
 * nutritional comparisons.
 */
import { useState, type FormEvent } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { Leaf, Sparkles, ArrowRight } from 'lucide-react'
import { Header } from '@/components/layout/header'
import { cn } from '@/lib/utils'

const SWAP_SUGGESTIONS = [
  { from: 'White Rice', icon: '🍚' },
  { from: 'Butter Naan', icon: '🫓' },
  { from: 'Samosa', icon: '🥟' },
  { from: 'Gulab Jamun', icon: '🍮' },
  { from: 'Puri Bhaji', icon: '🥘' },
  { from: 'Fried Pakoda', icon: '🧆' },
]

export default function HealthySwapsPage() {
  const navigate = useNavigate()
  const [dish, setDish] = useState('')

  const canSubmit = dish.trim().length > 1

  const handleSubmit = (e?: FormEvent) => {
    e?.preventDefault()
    if (!canSubmit) return
    const query = `What are healthy swaps for ${dish.trim()}? Compare nutrition and suggest alternatives.`
    navigate('/results', { state: { query } })
  }

  const applyQuick = (name: string) => {
    setDish(name)
  }

  return (
    <div className="min-h-screen bg-[var(--color-bg)]">
      <Header />
      <main className="mx-auto max-w-screen-2xl px-4 sm:px-6 lg:px-16 py-14">
        <div className="flex gap-12 xl:gap-16 items-start">
          {/* ── Left: form column ── */}
          <div className="flex-1 min-w-0">
        {/* Heading */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-10 text-center"
        >
          <div className="mb-3 flex justify-center">
            <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-green-500/10 border border-green-500/20">
              <Leaf size={22} className="text-green-500" />
            </div>
          </div>
          <p className="text-xs font-semibold uppercase tracking-widest text-green-500 mb-2">
            Healthy Swaps
          </p>
          <h1 className="text-3xl font-bold text-[var(--color-text)] mb-3">
            Find lighter alternatives
          </h1>
          <p className="text-sm text-[var(--color-text-muted)] max-w-sm mx-auto leading-relaxed">
            Enter any dish and the AI will recommend nutritionally better
            swaps, along with a side-by-side comparison.
          </p>
        </motion.div>

        {/* Quick dish buttons */}
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.08 }}
          className="mb-6 grid grid-cols-3 gap-2"
        >
          {SWAP_SUGGESTIONS.map(({ from, icon }) => (
            <button
              key={from}
              onClick={() => applyQuick(from)}
              className={cn(
                'rounded-xl border px-3 py-2.5 text-left transition-all',
                dish === from
                  ? 'border-green-500/50 bg-green-500/10'
                  : 'border-[var(--color-border)] bg-[var(--color-surface)] hover:border-green-500/30',
              )}
            >
              <span className="text-base leading-none">{icon}</span>
              <p className="mt-1 text-xs font-medium text-[var(--color-text)] leading-tight">
                {from}
              </p>
            </button>
          ))}
        </motion.div>

        {/* Input form */}
        <motion.form
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.14 }}
          onSubmit={handleSubmit}
          className="flex flex-col gap-4"
        >
          <div className="flex gap-2">
            <input
              type="text"
              value={dish}
              onChange={(e) => setDish(e.target.value)}
              placeholder="Enter any dish… e.g. Biryani, Pav Bhaji"
              className={cn(
                'flex-1 rounded-2xl border border-[var(--color-border)] bg-[var(--color-surface)]',
                'px-4 py-3 text-sm text-[var(--color-text)] placeholder-[var(--color-text-muted)]',
                'outline-none focus:border-green-500/50 transition-colors',
              )}
            />
            <button
              type="submit"
              disabled={!canSubmit}
              className={cn(
                'shrink-0 flex h-11 w-11 items-center justify-center rounded-2xl transition-all',
                canSubmit
                  ? 'bg-green-500 text-white hover:bg-green-600 shadow-lg shadow-green-500/20'
                  : 'bg-[var(--color-border)] text-[var(--color-text-muted)] cursor-not-allowed',
              )}
            >
              <ArrowRight size={16} />
            </button>
          </div>

          <button
            type="submit"
            disabled={!canSubmit}
            className={cn(
              'inline-flex w-full items-center justify-center gap-2 rounded-full py-3 text-sm font-semibold transition-all',
              canSubmit
                ? 'bg-green-500 text-white hover:bg-green-600 shadow-lg shadow-green-500/20'
                : 'bg-[var(--color-border)] text-[var(--color-text-muted)] cursor-not-allowed',
            )}
          >
            <Sparkles size={16} />
            Find Healthy Swaps
          </button>
        </motion.form>

        {/* Separator + search tip */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.3 }}
          className="mt-10 flex items-center gap-3"
        >
          <div className="flex-1 h-px bg-[var(--color-border)]" />
          <span className="text-xs text-[var(--color-text-muted)]">or</span>
          <div className="flex-1 h-px bg-[var(--color-border)]" />
        </motion.div>

        <motion.button
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.35 }}
          onClick={() =>
            navigate('/search?q=healthy+food&healthTags=Low+Calorie')
          }
          className="mt-4 flex w-full items-center justify-center gap-2 rounded-2xl border border-[var(--color-border)] bg-[var(--color-surface)] px-4 py-3 text-sm text-[var(--color-text-muted)] hover:text-[var(--color-text)] hover:border-green-500/30 transition-colors"
        >
          <Leaf size={14} className="text-green-500" />
          Browse low-calorie dishes in the graph database
          <ArrowRight size={13} />
        </motion.button>
          </div>{/* end form column */}

          {/* ── Right: desktop info panel ── */}
          <aside className="hidden xl:flex flex-col gap-5 w-80 flex-shrink-0 pt-16">
            <div className="rounded-2xl border border-[var(--color-border)] bg-[var(--color-surface)] p-5">
              <h3 className="text-xs font-semibold uppercase tracking-widest text-green-500 mb-4">What makes a healthy swap?</h3>
              <ul className="space-y-2.5">
                {[
                  'Lower calorie density per serving',
                  'Higher fibre for better satiety',
                  'Reduced saturated fat',
                  'Lower glycaemic index',
                  'More micronutrients (iron, calcium, vitamins)',
                  'Less sodium and free sugars',
                ].map((item) => (
                  <li key={item} className="flex items-start gap-2.5 text-sm">
                    <span className="mt-1.5 h-1.5 w-1.5 rounded-full flex-shrink-0 bg-green-500 opacity-70" />
                    <span className="text-[var(--color-text-muted)] leading-snug">{item}</span>
                  </li>
                ))}
              </ul>
            </div>
            <div className="rounded-2xl border border-[var(--color-border)] bg-[var(--color-surface)] p-5">
              <h3 className="text-xs font-semibold uppercase tracking-widest text-[var(--color-text-muted)] mb-4">Popular swaps</h3>
              <div className="space-y-2">
                {[
                  { from: 'White Rice', to: 'Brown Rice / Quinoa' },
                  { from: 'Puri', to: 'Chapati / Tandoori Roti' },
                  { from: 'Fried Samosa', to: 'Baked Samosa / Dhokla' },
                  { from: 'Gulab Jamun', to: 'Fruit Chaat' },
                  { from: 'Cream Curry', to: 'Tomato-based Curry' },
                ].map(({ from, to }) => (
                  <div key={from} className="flex items-center gap-2 text-xs">
                    <span className="text-[var(--color-text-muted)]">{from}</span>
                    <span className="text-green-500">→</span>
                    <span className="font-medium text-[var(--color-text)]">{to}</span>
                  </div>
                ))}
              </div>
            </div>
          </aside>
        </div>{/* end flex row */}
      </main>
    </div>
  )
}
