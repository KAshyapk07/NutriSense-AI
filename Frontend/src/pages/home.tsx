import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { Menu, Camera } from 'lucide-react'
import { SearchBar } from '@/components/ui/search-bar'
import { Logo } from '@/components/ui/logo'
import { SteamEffect } from '@/components/ui/steam-effect'
import { ImageDropZone } from '@/components/ui/image-drop-zone'
import { useSidebar } from '@/hooks/use-sidebar'
import landingBg from '@/assets/landing-bg.png'

const CHIPS = [
  { label: 'Sambar', query: 'Sambar', mode: 'search' },
  { label: 'Compare Dosa vs Idli', query: 'Compare Dosa vs Idli', mode: 'chat' },
  { label: 'Vegan Butter Chicken', query: 'Vegan Butter Chicken', mode: 'chat' },
  { label: 'Low calorie Dal', query: 'Low calorie Dal', mode: 'search' },
]

export default function Home() {
  const navigate = useNavigate()
  const { openSidebar } = useSidebar()
  const [showDropZone, setShowDropZone] = useState(false)
  const [pendingImage, setPendingImage] = useState<File | null>(null)

  const handleSearch = (query: string, image?: File) => {
    const img = image ?? pendingImage ?? undefined
    if (img) {
      navigate('/results', { state: { query, image: img } })
    } else {
      navigate(`/search?q=${encodeURIComponent(query)}`)
    }
  }

  const handleImageSelect = (file: File) => {
    setPendingImage(file)
    // Navigate immediately for image-only analysis
    navigate('/results', { state: { query: '', image: file } })
  }

  return (
    <div className="relative min-h-screen w-full overflow-hidden">

      {/* ── Hamburger — top-left, opens sidebar ── */}
      <button
        onClick={openSidebar}
        aria-label="Open navigation"
        className="absolute top-4 left-4 z-30 flex h-10 w-10 items-center justify-center rounded-lg
          text-white/60 hover:text-white hover:bg-white/10 transition-colors duration-150"
      >
        <Menu size={22} strokeWidth={1.75} />
      </button>

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

        {/* Image upload toggle */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.75, duration: 0.5 }}
          className="mt-3 w-full max-w-2xl"
        >
          <button
            onClick={() => setShowDropZone((v) => !v)}
            className="flex items-center gap-1.5 rounded-full border border-white/20 bg-white/8 backdrop-blur-md px-3.5 py-1.5 text-xs font-medium text-white/55 hover:bg-white/15 hover:text-white/75 hover:border-white/35 transition-all duration-200 mx-auto"
          >
            <Camera size={13} strokeWidth={1.5} />
            {showDropZone ? 'Hide image upload' : 'Analyse a food photo'}
          </button>

          <AnimatePresence>
            {showDropZone && (
              <motion.div
                initial={{ opacity: 0, height: 0, marginTop: 0 }}
                animate={{ opacity: 1, height: 'auto', marginTop: 12 }}
                exit={{ opacity: 0, height: 0, marginTop: 0 }}
                transition={{ duration: 0.3, ease: 'easeInOut' }}
                className="overflow-hidden"
              >
                <div className="rounded-2xl overflow-hidden border border-white/15 backdrop-blur-xl bg-black/30">
                  <ImageDropZone
                    onImageSelect={handleImageSelect}
                    onClear={() => setPendingImage(null)}
                    variant="full"
                  />
                </div>
              </motion.div>
            )}
          </AnimatePresence>
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
              onClick={() =>
                chip.mode === 'chat'
                  ? navigate('/results', { state: { query: chip.query } })
                  : handleSearch(chip.query)
              }
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
      <div className="absolute bottom-0 right-0 z-30 w-72 h-28
        bg-gradient-to-tl from-black from-50% via-black/95 via-70% to-transparent" />
      <div className="absolute bottom-0 right-0 z-31 w-48 h-16 bg-black" />

    </div>
  )
}
