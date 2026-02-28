/**
 * Image Analysis page — upload a photo, AI identifies and analyses the dish.
 * Route: /image
 */
import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { Sparkles } from 'lucide-react'
import { Header } from '@/components/layout/header'
import { ImageDropZone } from '@/components/ui/image-drop-zone'
import { cn } from '@/lib/utils'

export default function ImagePage() {
  const navigate = useNavigate()
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)

  const handleImageSelect = (file: File) => {
    setSelectedFile(file)
    setPreviewUrl(URL.createObjectURL(file))
  }

  const handleClear = () => {
    setSelectedFile(null)
    setPreviewUrl(null)
  }

  const handleAnalyse = () => {
    if (!selectedFile) return
    navigate('/results', {
      state: { query: '', image: selectedFile },
    })
  }

  return (
    <div className="min-h-screen bg-[var(--color-bg)]">
      <Header />
      <main className="mx-auto max-w-5xl px-4 sm:px-6 lg:px-12 py-14">
        {/* Page heading */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-10 text-center"
        >
          <p className="text-xs font-semibold uppercase tracking-widest text-[var(--color-accent)] mb-2">
            Image Analysis
          </p>
          <h1 className="text-3xl font-bold text-[var(--color-text)] mb-3">
            Upload a food photo
          </h1>
          <p className="text-sm text-[var(--color-text-muted)] max-w-sm mx-auto leading-relaxed">
            Drop any image of a dish — the AI will identify it and return a full
            nutritional breakdown with ingredients and instructions.
          </p>
        </motion.div>

        {/* Drop zone */}
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
        >
          <ImageDropZone
            onImageSelect={handleImageSelect}
            onClear={handleClear}
            previewUrl={previewUrl}
            variant="full"
          />
        </motion.div>

        {/* Analyse button */}
        {selectedFile && (
          <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            className="mt-6 flex justify-center"
          >
            <button
              onClick={handleAnalyse}
              className={cn(
                'inline-flex items-center gap-2 rounded-full px-8 py-3 text-sm font-semibold',
                'bg-[var(--color-accent)] text-[var(--color-accent-contrast)]',
                'hover:opacity-90 transition-opacity shadow-lg shadow-[var(--color-accent)]/20',
              )}
            >
              <Sparkles size={16} />
              Analyse Dish
            </button>
          </motion.div>
        )}

        {/* Helper tips */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.25 }}
          className="mt-10 grid grid-cols-3 gap-4"
        >
          {[
            { label: 'Good lighting', tip: 'Bright, even light gives better results' },
            { label: 'Single dish', tip: 'One food item per photo works best' },
            { label: 'Max 10 MB', tip: 'JPEG · PNG · WebP accepted' },
          ].map(({ label, tip }) => (
            <div
              key={label}
              className="rounded-xl border border-[var(--color-border)] bg-[var(--color-surface)] px-4 py-3 text-center"
            >
              <p className="text-xs font-semibold text-[var(--color-text)] mb-0.5">{label}</p>
              <p className="text-[11px] text-[var(--color-text-muted)] leading-snug">{tip}</p>
            </div>
          ))}
        </motion.div>
      </main>
    </div>
  )
}
