import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { SearchBar } from '@/components/ui/search-bar'
import { Logo } from '@/components/ui/logo'
import { SteamEffect } from '@/components/ui/steam-effect'
import landingBg from '@/assets/landing-bg.png'

const CHIPS = [
  { label: 'Sambar', query: 'Sambar' },
  { label: 'Compare Dosa vs Idli', query: 'Compare Dosa vs Idli' },
  { label: 'Vegan Butter Chicken', query: 'Vegan Butter Chicken' },
  { label: 'Low calorie Dal', query: 'Low calorie Dal' },
]

export default function Home() {
  const navigate = useNavigate()

  const handleSearch = (query: string, image?: File) => {
    navigate('/results', { state: { query, image } })
  }

  return (
    <div className="relative min-h-screen w-full overflow-hidden">

      {/* ── Background image — full bleed, high visibility ── */}
      <div
        className="absolute inset-0 z-0"
        style={{
          backgroundImage: `url(${landingBg})`,
          backgroundSize: 'cover',
          backgroundPosition: 'center center',
          backgroundRepeat: 'no-repeat',
        }}
      />

      {/* Top and bottom dark vignette for text readability — preserves center image */}
      <div className="absolute inset-0 z-[1] bg-gradient-to-b
        from-black/80 via-black/25 to-black/65" />

      {/* ── Live steam / smoke canvas ── */}
      <SteamEffect />

      {/* ── Main content — slightly left of center to balance the bowl ── */}
      <div className="relative z-20 flex min-h-screen flex-col items-center justify-center px-4 py-16 lg:pr-[22vw]">

        {/* Logo */}
        <motion.div
          initial={{ opacity: 0, y: -32 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
          className="text-center"
        >
          <Logo
            size="xl"
            alwaysWhite
            className="drop-shadow-2xl"
          />
        </motion.div>

        {/* Tagline */}
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.35, duration: 0.6 }}
          className="mt-4 text-[11px] font-sans font-medium uppercase tracking-[0.5em] text-white/65"
          style={{ textShadow: '0 1px 10px rgba(0,0,0,0.8)' }}
        >
          Intelligent Nutrition Analysis
        </motion.p>

        {/* Description */}
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5, duration: 0.6 }}
          className="mt-5 max-w-lg text-center text-sm text-white/50 leading-relaxed font-sans"
          style={{ textShadow: '0 1px 8px rgba(0,0,0,0.9)' }}
        >
          Search any dish for nutrition facts · Compare two dishes side-by-side ·
          Modify recipes for your dietary needs · Upload a food photo for instant analysis
        </motion.p>

        {/* Search bar — wide */}
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6, duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
          className="mt-10 w-full max-w-2xl"
        >
          <SearchBar
            onSearch={handleSearch}
            autoFocus
            glass
            placeholder="Search any dish, compare, or modify..."
          />
        </motion.div>

        {/* Suggestion chips */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.85, duration: 0.5 }}
          className="mt-5 flex flex-wrap justify-center gap-2"
        >
          {CHIPS.map((chip, i) => (
            <motion.button
              key={chip.label}
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.95 + i * 0.07, duration: 0.3 }}
              onClick={() => handleSearch(chip.query)}
              className="rounded-full border border-white/20 bg-white/10 backdrop-blur-md
                px-4 py-1.5 text-xs font-medium text-white/65
                hover:bg-white/18 hover:text-white hover:border-white/40
                transition-all duration-200"
            >
              {chip.label}
            </motion.button>
          ))}
        </motion.div>
      </div>

      {/* ── Corner watermark blocker (bottom-right) ── */}
      <div className="absolute bottom-0 right-0 z-30 w-52 h-20
        bg-gradient-to-tl from-black from-40% via-black/95 via-60% to-transparent" />

    </div>
  )
}
