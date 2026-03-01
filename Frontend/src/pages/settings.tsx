import { useState } from 'react'
import { motion } from 'framer-motion'
import { useNavigate } from 'react-router-dom'
import {
  Sun,
  Moon,
  Leaf,
  AlertTriangle,
  Info,
  ChevronRight,
  Check,
  Cpu,
  Database,
  Globe,
  Bell,
  Shield,
  Github,
} from 'lucide-react'
import { Header } from '@/components/layout/header'
import { useTheme } from '@/hooks/use-theme'
import { usePreferences } from '@/hooks/use-preferences'
import { cn } from '@/lib/utils'

// ─── types ──────────────────────────────────────────────────────────────────
type DietTag = {
  id: string
  label: string
  description: string
}

type AllergenTag = {
  id: string
  label: string
  emoji: string
}

// ─── data ────────────────────────────────────────────────────────────────────
const DIET_TAGS: DietTag[] = [
  { id: 'vegetarian', label: 'Vegetarian', description: 'Exclude meat & seafood' },
  { id: 'vegan', label: 'Vegan', description: 'Exclude all animal products' },
  { id: 'gluten-free', label: 'Gluten-Free', description: 'Exclude wheat, barley, rye' },
  { id: 'diabetic-friendly', label: 'Diabetic Friendly', description: 'Low glycaemic index dishes' },
  { id: 'high-protein', label: 'High Protein', description: 'Prioritise protein-dense results' },
  { id: 'low-calorie', label: 'Low Calorie', description: 'Under 400 kcal per serving' },
  { id: 'heart-healthy', label: 'Heart Healthy', description: 'Low sodium & saturated fat' },
  { id: 'keto', label: 'Keto', description: 'Very low carbohydrate dishes' },
]

const ALLERGENS: AllergenTag[] = [
  { id: 'dairy', label: 'Dairy', emoji: '' },
  { id: 'nuts', label: 'Nuts', emoji: '' },
  { id: 'gluten', label: 'Gluten', emoji: '' },
  { id: 'soy', label: 'Soy', emoji: '' },
  { id: 'eggs', label: 'Eggs', emoji: '' },
  { id: 'shellfish', label: 'Shellfish', emoji: '' },
  { id: 'sesame', label: 'Sesame', emoji: '' },
  { id: 'mustard', label: 'Mustard', emoji: '' },
]

const fadeUp = {
  hidden: { opacity: 0, y: 14 },
  show: (i: number) => ({
    opacity: 1,
    y: 0,
    transition: { delay: i * 0.06, duration: 0.4, ease: [0.22, 1, 0.36, 1] },
  }),
}

// ─── reusable row wrapper ─────────────────────────────────────────────────────
function SettingsSection({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section>
      <p className="text-xs font-semibold uppercase tracking-[0.18em] text-[var(--color-text-muted)] mb-2 px-1">
        {title}
      </p>
      <div className="rounded-2xl bg-[var(--color-surface)] border border-[var(--color-border)] overflow-hidden divide-y divide-[var(--color-border)]">
        {children}
      </div>
    </section>
  )
}

function SettingsRow({
  icon: Icon,
  label,
  description,
  right,
  onClick,
}: {
  icon: React.FC<{ size?: number; className?: string; strokeWidth?: number }>
  label: string
  description?: string
  right?: React.ReactNode
  onClick?: () => void
}) {
  return (
    <button
      onClick={onClick}
      disabled={!onClick}
      className={cn(
        'w-full flex items-center gap-4 px-4 py-3.5 text-left',
        onClick ? 'hover:bg-[var(--color-bg)] transition-colors duration-150' : 'cursor-default',
      )}
    >
      <Icon size={16} strokeWidth={1.75} className="flex-shrink-0 text-[var(--color-text-muted)]" />
      <div className="flex-1 min-w-0">
        <p className="text-sm font-medium text-[var(--color-text)]">{label}</p>
        {description && (
          <p className="text-xs text-[var(--color-text-muted)] mt-0.5 leading-snug">{description}</p>
        )}
      </div>
      {right ?? (onClick && <ChevronRight size={14} className="flex-shrink-0 text-[var(--color-text-muted)]" />)}
    </button>
  )
}

// ─── main page ────────────────────────────────────────────────────────────────
export default function Settings() {
  const navigate = useNavigate()
  const { theme, toggle } = useTheme()
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [notifications, setNotifications] = useState(() => {
    if (typeof window === 'undefined') return false
    return localStorage.getItem('nutrisense-notifications') === 'true'
  })

  // Persist notification preference
  const toggleNotifications = () => {
    setNotifications((v) => {
      const next = !v
      localStorage.setItem('nutrisense-notifications', String(next))
      return next
    })
  }

  // Persistent preferences
  const { prefs, setActiveDiets, setExcludeAllergens, clearAll } = usePreferences()
  const activeDiets = new Set(prefs.activeDiets)
  const activeAllergens = new Set(prefs.excludeAllergens)

  const toggleDiet = (id: string) => {
    const next = new Set(activeDiets)
    next.has(id) ? next.delete(id) : next.add(id)
    setActiveDiets([...next])
  }

  const toggleAllergen = (id: string) => {
    const next = new Set(activeAllergens)
    next.has(id) ? next.delete(id) : next.add(id)
    setExcludeAllergens([...next])
  }

  return (
    <div className="min-h-screen bg-[var(--color-bg)]">

      <Header />

      <main className="max-w-screen-xl mx-auto px-4 sm:px-6 lg:px-16 py-8 space-y-7">

        {/* ── Appearance ── */}
        <motion.div variants={fadeUp} custom={0} initial="hidden" animate="show">
          <SettingsSection title="Appearance">
            <SettingsRow
              icon={theme === 'dark' ? Moon : Sun}
              label="Theme"
              description={theme === 'dark' ? 'Dark mode active' : 'Light mode active'}
              onClick={toggle}
              right={
                <div className="flex items-center gap-2">
                  <Sun size={13} className="text-[var(--color-text-muted)]" />
                  <button
                    onClick={(e) => { e.stopPropagation(); toggle() }}
                    className={cn(
                      'relative h-6 w-11 rounded-full border transition-colors duration-200',
                      theme === 'dark'
                        ? 'bg-[var(--color-accent)] border-[var(--color-accent)]'
                        : 'bg-[var(--color-bg)] border-[var(--color-border)]',
                    )}
                    aria-label="Toggle theme"
                  >
                    <span className={cn(
                      'absolute top-0.5 left-0.5 h-5 w-5 rounded-full transition-transform duration-200',
                      'bg-[var(--color-accent-contrast)] shadow-sm',
                      theme === 'dark' ? 'translate-x-5' : 'translate-x-0',
                    )} />
                  </button>
                  <Moon size={13} className="text-[var(--color-text-muted)]" />
                </div>
              }
            />
          </SettingsSection>
        </motion.div>

        {/* ── Dietary Preferences ── */}
        <motion.div variants={fadeUp} custom={1} initial="hidden" animate="show">
          <SettingsSection title="Dietary Preferences">
            <div className="px-4 py-3">
              <p className="text-xs text-[var(--color-text-muted)] mb-3 leading-relaxed">
                Selected tags are used to pre-filter search results and recipe suggestions.
                Requires sign-in to persist across sessions.
              </p>
              <div className="flex flex-wrap gap-2">
                {DIET_TAGS.map((tag) => {
                  const active = activeDiets.has(tag.id)
                  return (
                    <button
                      key={tag.id}
                      onClick={() => toggleDiet(tag.id)}
                      title={tag.description}
                      className={cn(
                        'flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium',
                        'border transition-all duration-150',
                        active
                          ? 'bg-[var(--color-accent)] text-[var(--color-accent-contrast)] border-[var(--color-accent)]'
                          : 'bg-[var(--color-bg)] text-[var(--color-text)] border-[var(--color-border)] hover:border-[var(--color-text-muted)]',
                      )}
                    >
                      {active && <Check size={11} strokeWidth={2.5} />}
                      <Leaf size={11} strokeWidth={1.75} />
                      {tag.label}
                    </button>
                  )
                })}
              </div>
            </div>
          </SettingsSection>
        </motion.div>

        {/* ── Allergen Profile ── */}
        <motion.div variants={fadeUp} custom={2} initial="hidden" animate="show">
          <SettingsSection title="Allergen Profile">
            <div className="px-4 py-3">
              <p className="text-xs text-[var(--color-text-muted)] mb-3 leading-relaxed">
                Dishes containing your allergens are silently excluded from all results.
                You'll never see them, even without an active filter.
              </p>
              <div className="flex flex-wrap gap-2">
                {ALLERGENS.map((allergen) => {
                  const active = activeAllergens.has(allergen.id)
                  return (
                    <button
                      key={allergen.id}
                      onClick={() => toggleAllergen(allergen.id)}
                      className={cn(
                        'flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium',
                        'border transition-all duration-150',
                        active
                          ? 'bg-red-500/10 text-red-500 border-red-500/40 dark:text-red-400 dark:border-red-400/40'
                          : 'bg-[var(--color-bg)] text-[var(--color-text)] border-[var(--color-border)] hover:border-[var(--color-text-muted)]',
                      )}
                    >
                      {allergen.label}
                      {active && (
                        <span className="flex h-4 w-4 items-center justify-center rounded-full
                          bg-red-500/20">
                          <AlertTriangle size={9} strokeWidth={2} />
                        </span>
                      )}
                    </button>
                  )
                })}
              </div>
              {activeAllergens.size > 0 && (
                <motion.p
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  className="mt-3 text-[11px] text-red-500 dark:text-red-400 flex items-center gap-1.5"
                >
                  <AlertTriangle size={11} strokeWidth={2} />
                  {activeAllergens.size} allergen{activeAllergens.size > 1 ? 's' : ''} selected.
                  Sign in to persist this across sessions.
                </motion.p>
              )}
            </div>
          </SettingsSection>
        </motion.div>

        {/* ── Notifications ── */}
        <motion.div variants={fadeUp} custom={3} initial="hidden" animate="show">
          <SettingsSection title="Notifications">
            <SettingsRow
              icon={Bell}
              label="Weekly Nutrition Digest"
              description="A summary of your most searched dishes and insights"
              onClick={toggleNotifications}
              right={
                <button
                  onClick={(e) => { e.stopPropagation(); toggleNotifications() }}
                  className={cn(
                    'relative h-6 w-11 rounded-full border transition-colors duration-200',
                    notifications
                      ? 'bg-[var(--color-accent)] border-[var(--color-accent)]'
                      : 'bg-[var(--color-bg)] border-[var(--color-border)]',
                  )}
                  aria-label="Toggle notifications"
                >
                  <span className={cn(
                    'absolute top-0.5 left-0.5 h-5 w-5 rounded-full transition-transform duration-200',
                    'bg-[var(--color-accent-contrast)] shadow-sm',
                    notifications ? 'translate-x-5' : 'translate-x-0',
                  )} />
                </button>
              }
            />
          </SettingsSection>
        </motion.div>

        {/* ── About ── */}
        <motion.div variants={fadeUp} custom={4} initial="hidden" animate="show">
          <SettingsSection title="About">
            <SettingsRow
              icon={Cpu}
              label="AI Model"
              description="Llama 3.2 via Ollama (local, no API cost)"
            />
            <SettingsRow
              icon={Database}
              label="Knowledge Graph"
              description="Neo4j · 725+ Indian dishes · Dual food cluster"
            />
            <SettingsRow
              icon={Globe}
              label="Image Classifier"
              description="EfficientNet-B4 · Trained on Indian food dataset"
            />
            <SettingsRow
              icon={Shield}
              label="Privacy"
              description="All data stays local — no external API calls for food data"
            />
            <SettingsRow
              icon={Info}
              label="Version"
              description="NutriSense AI · Phase 4 · February 2026"
            />
            <SettingsRow
              icon={Github}
              label="Source Code"
              description="github.com/NutriSense-AI"
              onClick={() => window.open('https://github.com', '_blank')}
            />
          </SettingsSection>
        </motion.div>

        {/* ── Danger zone ── */}
        <motion.div variants={fadeUp} custom={5} initial="hidden" animate="show">
          <SettingsSection title="Data">
            <SettingsRow
              icon={AlertTriangle}
              label="Clear Search History"
              description="Remove all locally stored search queries and preferences"
              onClick={() => clearAll()}
              right={
                <span className="text-xs text-[var(--color-text-muted)] px-2 py-0.5
                  rounded-full border border-[var(--color-border)]">
                  Local only
                </span>
              }
            />
          </SettingsSection>
        </motion.div>

        <div className="pb-8" />
      </main>
    </div>
  )
}
