import { motion } from 'framer-motion'
import { useNavigate } from 'react-router-dom'
import {
  User,
  LogIn,
  Clock,
  Heart,
  ChefHat,
  Search,
  Camera,
  GitCompare,
  Sliders,
  Star,
  TrendingUp,
  Lock,
} from 'lucide-react'
import { Header } from '@/components/layout/header'
import { cn } from '@/lib/utils'

// ─── mock recent searches shown in guest state ─────────────────────────────
const RECENT_SEARCHES = [
  { query: 'Butter Chicken', type: 'text', time: '2 hours ago', icon: Search },
  { query: 'Food photo', type: 'image', time: 'Yesterday', icon: Camera },
  { query: 'Compare Dosa vs Idli', type: 'compare', time: '3 days ago', icon: GitCompare },
  { query: 'Low-calorie Dal Makhani', type: 'modify', time: '5 days ago', icon: Sliders },
]

const STATS = [
  { label: 'Searches', value: '—', icon: Search, locked: true },
  { label: 'Saved Recipes', value: '—', icon: Heart, locked: true },
  { label: 'Dishes Cooked', value: '—', icon: ChefHat, locked: true },
  { label: 'Streak', value: '—', icon: TrendingUp, locked: true },
]

const CAPABILITIES = [
  {
    icon: Clock,
    title: 'Search History',
    description: 'Every query saved and searchable. Revisit past results instantly.',
  },
  {
    icon: Heart,
    title: 'Saved Recipes',
    description: 'Bookmark dishes you love. Build your personal recipe library.',
  },
  {
    icon: Star,
    title: 'Personalised Recommendations',
    description: "AI learns your taste profile and suggests dishes you'll actually enjoy.",
  },
  {
    icon: Lock,
    title: 'Allergen Profile',
    description: "Set your allergens once — they're silently filtered from every result.",
  },
]

const fadeUp = {
  hidden: { opacity: 0, y: 18 },
  show: (i: number) => ({
    opacity: 1,
    y: 0,
    transition: { delay: i * 0.07, duration: 0.45, ease: [0.22, 1, 0.36, 1] },
  }),
}

export default function Profile() {
  const navigate = useNavigate()

  return (
    <div className="min-h-screen bg-[var(--color-bg)]">

      <Header />

      <main className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-12 py-8 space-y-8">

        {/* ── Avatar + guest banner ── */}
        <motion.section
          variants={fadeUp} custom={0} initial="hidden" animate="show"
          className="flex flex-col items-center text-center gap-4"
        >
          <div className="relative">
            <div className="flex h-20 w-20 items-center justify-center rounded-full
              bg-[var(--color-surface)] border-2 border-[var(--color-border)] shadow-md">
              <User size={34} className="text-[var(--color-text-muted)]" />
            </div>
            <span className="absolute -bottom-1 -right-1 flex h-6 w-6 items-center justify-center
              rounded-full bg-[var(--color-bg)] border border-[var(--color-border)]">
              <Lock size={11} className="text-[var(--color-text-muted)]" />
            </span>
          </div>

          <div>
            <p className="text-lg font-semibold text-[var(--color-text)]">Guest User</p>
            <p className="text-sm text-[var(--color-text-muted)] mt-0.5">Not signed in</p>
          </div>

          {/* Sign-in CTA */}
          <button
            className={cn(
              'flex items-center gap-2 px-6 py-2.5 rounded-full text-sm font-medium',
              'bg-[var(--color-accent)] text-[var(--color-accent-contrast)]',
              'hover:opacity-90 transition-opacity duration-150',
            )}
          >
            <LogIn size={15} strokeWidth={2} />
            Sign in with Google
          </button>
          <p className="text-xs text-[var(--color-text-muted)] max-w-xs leading-relaxed">
            Sign in to unlock personalised recommendations, saved recipes, allergen filtering, and your full search history.
          </p>
        </motion.section>

        {/* ── Stats grid ── */}
        <motion.section variants={fadeUp} custom={1} initial="hidden" animate="show">
          <p className="text-xs font-semibold uppercase tracking-[0.15em] text-[var(--color-text-muted)] mb-3">
            Your Stats
          </p>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            {STATS.map((stat) => {
              const Icon = stat.icon
              return (
                <div
                  key={stat.label}
                  className="flex flex-col items-center gap-1.5 py-4 px-3 rounded-2xl
                    bg-[var(--color-surface)] border border-[var(--color-border)] text-center"
                >
                  <Icon size={18} className="text-[var(--color-text-muted)]" strokeWidth={1.5} />
                  <p className={cn(
                    'text-xl font-bold tracking-tight',
                    stat.locked ? 'text-[var(--color-text-muted)]' : 'text-[var(--color-text)]',
                  )}>
                    {stat.value}
                  </p>
                  <p className="text-[11px] text-[var(--color-text-muted)]">{stat.label}</p>
                </div>
              )
            })}
          </div>
        </motion.section>

        {/* ── Recent searches (demo) ── */}
        <motion.section variants={fadeUp} custom={2} initial="hidden" animate="show">
          <div className="flex items-center justify-between mb-3">
            <p className="text-xs font-semibold uppercase tracking-[0.15em] text-[var(--color-text-muted)]">
              Recent Searches
            </p>
            <span className="text-xs text-[var(--color-text-muted)] bg-[var(--color-bg)]
              border border-[var(--color-border)] px-2 py-0.5 rounded-full">
              Demo only
            </span>
          </div>
          <ul className="space-y-2">
            {RECENT_SEARCHES.map((item, i) => {
              const Icon = item.icon
              return (
                <motion.li
                  key={i}
                  variants={fadeUp} custom={3 + i} initial="hidden" animate="show"
                >
                  <div className="flex items-center gap-3 px-4 py-3 rounded-xl
                    bg-[var(--color-surface)] border border-[var(--color-border)]">
                    <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-[var(--color-bg)]">
                      <Icon size={15} className="text-[var(--color-text-muted)]" strokeWidth={1.75} />
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium text-[var(--color-text)] truncate">{item.query}</p>
                      <p className="text-[11px] text-[var(--color-text-muted)]">{item.time}</p>
                    </div>
                    <Lock size={12} className="flex-shrink-0 text-[var(--color-text-muted)] opacity-50" />
                  </div>
                </motion.li>
              )
            })}
          </ul>
        </motion.section>

        {/* ── What you unlock ── */}
        <motion.section variants={fadeUp} custom={7} initial="hidden" animate="show">
          <p className="text-xs font-semibold uppercase tracking-[0.15em] text-[var(--color-text-muted)] mb-3">
            Unlock with an account
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {CAPABILITIES.map((cap, i) => {
              const Icon = cap.icon
              return (
                <motion.div
                  key={i}
                  variants={fadeUp} custom={8 + i} initial="hidden" animate="show"
                  className="flex gap-3 px-4 py-3.5 rounded-2xl
                    bg-[var(--color-surface)] border border-[var(--color-border)]"
                >
                  <div className="flex h-9 w-9 flex-shrink-0 items-center justify-center rounded-xl bg-[var(--color-bg)]">
                    <Icon size={16} className="text-[var(--color-text-muted)]" strokeWidth={1.75} />
                  </div>
                  <div>
                    <p className="text-sm font-medium text-[var(--color-text)]">{cap.title}</p>
                    <p className="text-xs text-[var(--color-text-muted)] mt-0.5 leading-relaxed">{cap.description}</p>
                  </div>
                </motion.div>
              )
            })}
          </div>
        </motion.section>

        {/* ── Platform capabilities ── */}
        <motion.section variants={fadeUp} custom={12} initial="hidden" animate="show">
          <p className="text-xs font-semibold uppercase tracking-[0.15em] text-[var(--color-text-muted)] mb-3">
            What NutriSense supports
          </p>
          <div className="px-4 py-4 rounded-2xl bg-[var(--color-surface)] border border-[var(--color-border)] space-y-2.5">
            {[
              '725+ Indian dish & recipe database',
              'Packaged food product lookup',
              'Neo4j knowledge graph queries',
              'LLM-powered reasoning (Llama 3.2)',
              'EfficientNet-B4 image recognition',
              'Fuzzy recipe matching engine',
              'Allergen & health-tag filtering',
              'Multi-pathway query routing',
            ].map((cap) => (
              <div key={cap} className="flex items-start gap-2.5">
                <span className="mt-1.5 h-1.5 w-1.5 rounded-full flex-shrink-0 bg-[var(--color-accent)] opacity-70" />
                <span className="text-sm text-[var(--color-text-muted)] leading-snug">{cap}</span>
              </div>
            ))}
          </div>
        </motion.section>

        {/* ── Bottom sign-in CTA ── */}
        <motion.div
          variants={fadeUp} custom={12} initial="hidden" animate="show"
          className="flex flex-col items-center pb-6 gap-3"
        >
          <button
            className={cn(
              'flex items-center gap-2 px-8 py-3 rounded-full text-sm font-medium w-full justify-center',
              'bg-[var(--color-accent)] text-[var(--color-accent-contrast)]',
              'hover:opacity-90 transition-opacity duration-150',
            )}
          >
            <LogIn size={15} strokeWidth={2} />
            Create free account
          </button>
          <p className="text-xs text-[var(--color-text-muted)]">
            Authentication coming in Phase 6 · Google OAuth · Free forever
          </p>
        </motion.div>

      </main>
    </div>
  )
}
