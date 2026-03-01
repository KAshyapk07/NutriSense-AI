import { useState, useRef, useCallback, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Menu,
  Camera,
  Search,
  ArrowRight,
  X,
  Sparkles,
  Flame,
  Leaf,
  Zap,
  UtensilsCrossed,
  ImageIcon,
} from 'lucide-react'
import { Logo } from '@/components/ui/logo'
import { SteamEffect } from '@/components/ui/steam-effect'
import { useSidebar } from '@/hooks/use-sidebar'
import { cn } from '@/lib/utils'
import landingBg from '@/assets/landing-bg.png'

/* ── Example suggestion chips — varied, showcasing all pathways ── */
const CHIPS = [
  { label: 'High protein breakfast', icon: Flame, mode: 'search' },
  { label: 'Compare Biryani vs Pulao', icon: Zap, mode: 'chat' },
  { label: 'Keto-friendly snacks', icon: Leaf, mode: 'search' },
  { label: 'Make Paneer Tikka low calorie', icon: Sparkles, mode: 'chat' },
  { label: 'Diabetic friendly desserts', icon: Leaf, mode: 'search' },
  { label: 'Sambar nutrition facts', icon: UtensilsCrossed, mode: 'chat' },
]

/* ── Accepted image types ── */
const ACCEPTED_TYPES = ['image/jpeg', 'image/png', 'image/webp', 'image/gif']

export default function Home() {
  const navigate = useNavigate()
  const { openSidebar } = useSidebar()

  /* ── Omnibox state ── */
  const [query, setQuery] = useState('')
  const [isFocused, setIsFocused] = useState(false)
  const [selectedImage, setSelectedImage] = useState<File | null>(null)
  const [imagePreviewUrl, setImagePreviewUrl] = useState<string | null>(null)
  const [isDragging, setIsDragging] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const dragCounter = useRef(0)

  /* ── Search logic — routes to /search for text, /results for images ── */
  const handleSubmit = useCallback(
    (q?: string) => {
      const text = (q ?? query).trim()
      if (selectedImage) {
        navigate('/results', { state: { query: text, image: selectedImage } })
      } else if (text) {
        navigate(`/search?q=${encodeURIComponent(text)}`)
      }
    },
    [query, selectedImage, navigate],
  )

  const handleChipClick = (chip: (typeof CHIPS)[number]) => {
    if (chip.mode === 'chat') {
      navigate('/results', { state: { query: chip.label } })
    } else {
      navigate(`/search?q=${encodeURIComponent(chip.label)}`)
    }
  }

  /* ── Image handling ── */
  const handleImageFile = useCallback((file: File) => {
    if (!ACCEPTED_TYPES.includes(file.type)) return
    if (file.size > 10 * 1024 * 1024) return
    setSelectedImage(file)
    setImagePreviewUrl(URL.createObjectURL(file))
  }, [])

  const removeImage = useCallback(() => {
    setSelectedImage(null)
    if (imagePreviewUrl) URL.revokeObjectURL(imagePreviewUrl)
    setImagePreviewUrl(null)
    if (fileInputRef.current) fileInputRef.current.value = ''
  }, [imagePreviewUrl])

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) handleImageFile(file)
  }

  /* ── Drag and drop on the omnibox ── */
  const handleDragEnter = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    dragCounter.current++
    if (e.dataTransfer.types.includes('Files')) setIsDragging(true)
  }

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    dragCounter.current--
    if (dragCounter.current === 0) setIsDragging(false)
  }

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)
    dragCounter.current = 0
    const file = e.dataTransfer.files?.[0]
    if (file) handleImageFile(file)
  }

  /* ── Paste image from clipboard ── */
  useEffect(() => {
    const handlePaste = (e: ClipboardEvent) => {
      const items = e.clipboardData?.items
      if (!items) return
      for (const item of items) {
        if (item.kind === 'file' && ACCEPTED_TYPES.includes(item.type)) {
          const file = item.getAsFile()
          if (file) handleImageFile(file)
          break
        }
      }
    }
    window.addEventListener('paste', handlePaste)
    return () => window.removeEventListener('paste', handlePaste)
  }, [handleImageFile])

  const hasInput = query.trim() || selectedImage

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

      {/* Top and bottom dark vignette for text readability */}
      <div className="absolute inset-0 z-[1] bg-gradient-to-b
        from-black/80 via-black/25 to-black/65" />

      {/* ── Live steam / smoke canvas ── */}
      <SteamEffect />

      {/* ── Main content ── */}
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

        {/* ── Omnibox — Perplexity / Gemini style ── */}
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6, duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
          className="mt-10 w-full max-w-2xl"
        >
          <div
            onDragEnter={handleDragEnter}
            onDragLeave={handleDragLeave}
            onDragOver={handleDragOver}
            onDrop={handleDrop}
            className={cn(
              'relative rounded-2xl border backdrop-blur-xl transition-all duration-300',
              isDragging
                ? 'border-white/60 bg-white/20 shadow-[0_0_24px_rgba(255,255,255,0.15)]'
                : isFocused
                  ? 'border-white/50 bg-white/15 shadow-[0_0_0_1px_rgba(255,255,255,0.3),0_8px_32px_rgba(0,0,0,0.4)]'
                  : 'border-white/25 bg-white/10 hover:bg-white/14 hover:border-white/35',
            )}
          >
            {/* Drag overlay */}
            <AnimatePresence>
              {isDragging && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="absolute inset-0 z-10 rounded-2xl flex items-center justify-center bg-white/10 backdrop-blur-sm"
                >
                  <div className="flex items-center gap-2 text-white/80 text-sm font-medium">
                    <ImageIcon size={18} />
                    Drop image here
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Image preview row — inside the box */}
            <AnimatePresence>
              {imagePreviewUrl && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  className="overflow-hidden"
                >
                  <div className="flex items-center gap-3 px-4 pt-3">
                    <div className="relative h-16 w-16 rounded-xl overflow-hidden border border-white/20 flex-shrink-0">
                      <img
                        src={imagePreviewUrl}
                        alt="Upload preview"
                        className="h-full w-full object-cover"
                      />
                      <button
                        type="button"
                        onClick={removeImage}
                        className="absolute inset-0 flex items-center justify-center bg-black/50 opacity-0 hover:opacity-100 transition-opacity"
                      >
                        <X size={14} className="text-white" />
                      </button>
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-xs font-medium text-white/70 truncate">
                        {selectedImage?.name}
                      </p>
                      <p className="text-[11px] text-white/40">
                        {selectedImage
                          ? `${(selectedImage.size / 1024).toFixed(0)} KB`
                          : ''}
                      </p>
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Input row */}
            <form
              onSubmit={(e) => {
                e.preventDefault()
                handleSubmit()
              }}
              className="flex items-center gap-2 px-4 py-3"
            >
              <Search
                size={20}
                strokeWidth={isFocused ? 2.2 : 1.5}
                className={cn(
                  'flex-shrink-0 transition-all duration-300',
                  isFocused ? 'text-white' : 'text-white/50',
                )}
              />
              <input
                ref={inputRef}
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onFocus={() => setIsFocused(true)}
                onBlur={() => setTimeout(() => setIsFocused(false), 200)}
                placeholder="Ask anything about nutrition..."
                autoFocus
                className="flex-1 bg-transparent text-white placeholder:text-white/40 text-base outline-none font-sans"
              />
              {/* Hidden file input */}
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={handleFileInput}
                className="hidden"
              />
              {/* Camera button */}
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                className="flex-shrink-0 flex h-9 w-9 items-center justify-center rounded-lg text-white/50 hover:text-white hover:bg-white/10 transition-all duration-150"
                aria-label="Upload image"
              >
                <Camera size={18} strokeWidth={1.5} />
              </button>
              {/* Submit */}
              <AnimatePresence>
                {hasInput && (
                  <motion.button
                    type="submit"
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.8 }}
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    className="flex-shrink-0 flex h-9 w-9 items-center justify-center rounded-full bg-white text-black transition-shadow hover:shadow-lg"
                  >
                    <ArrowRight size={16} strokeWidth={2} />
                  </motion.button>
                )}
              </AnimatePresence>
            </form>

            {/* Bottom helper text */}
            <div className="flex items-center gap-3 px-4 pb-2.5 text-[11px] text-white/30">
              <span>Paste or drop an image</span>
              <span className="w-px h-3 bg-white/15" />
              <span>Enter to search</span>
            </div>
          </div>
        </motion.div>

        {/* ── Suggestion chips ── */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.85, duration: 0.5 }}
          className="mt-6 flex flex-wrap justify-center gap-2.5 max-w-2xl"
        >
          {CHIPS.map((chip, i) => (
            <motion.button
              key={chip.label}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.95 + i * 0.06, duration: 0.3 }}
              onClick={() => handleChipClick(chip)}
              className="group flex items-center gap-2 rounded-full border border-white/15 bg-white/8 backdrop-blur-md
                pl-3 pr-4 py-2 text-xs font-medium text-white/60
                hover:bg-white/16 hover:text-white hover:border-white/35
                transition-all duration-200"
            >
              <chip.icon
                size={13}
                strokeWidth={1.75}
                className="text-white/40 group-hover:text-white/80 transition-colors"
              />
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
