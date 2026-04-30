import { useState } from 'react'
import { AlertTriangle, MessageSquare, X, Send, Check } from 'lucide-react'
import { submitAiFeedback } from '@/lib/api'
import { cn } from '@/lib/utils'

interface AiResponseFooterProps {
  aiResponse: string
  context?: string
  dark?: boolean
}

export function AiResponseFooter({ aiResponse, context, dark }: AiResponseFooterProps) {
  const [open, setOpen] = useState(false)
  const [comment, setComment] = useState('')
  const [status, setStatus] = useState<'idle' | 'sending' | 'done' | 'error'>('idle')

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!comment.trim()) return
    setStatus('sending')
    try {
      await submitAiFeedback({ ai_response: aiResponse, user_comment: comment.trim(), context })
      setStatus('done')
      setTimeout(() => { setOpen(false); setComment(''); setStatus('idle') }, 1800)
    } catch {
      setStatus('error')
    }
  }

  const mutedClass = dark ? 'text-white/30' : 'text-[var(--color-text-muted)]'
  const hoverClass = dark ? 'hover:text-white/60' : 'hover:text-[var(--color-text)]'

  return (
    <>
      <div className="flex items-center justify-between mt-2 px-1">
        <span className={cn('inline-flex items-center gap-1 text-[10px]', mutedClass)}>
          <AlertTriangle size={9} strokeWidth={1.75} />
          AI-generated — verify before acting on it
        </span>
        <button
          onClick={() => setOpen(true)}
          className={cn(
            'inline-flex items-center gap-1.5 text-[11px] font-medium transition-colors duration-150',
            mutedClass, hoverClass,
          )}
          title="Leave feedback on this response"
        >
          <MessageSquare size={11} strokeWidth={1.75} />
          Feedback
        </button>
      </div>

      {open && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm"
          onClick={(e) => { if (e.target === e.currentTarget) setOpen(false) }}
        >
          <div className="w-full max-w-md rounded-2xl border border-[var(--color-border)] bg-[var(--color-surface)] p-6 shadow-2xl">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <MessageSquare size={15} strokeWidth={1.75} className="text-[var(--color-text-muted)]" />
                <h3 className="text-sm font-semibold text-[var(--color-text)]">Rate this response</h3>
              </div>
              <button
                onClick={() => setOpen(false)}
                className="text-[var(--color-text-muted)] hover:text-[var(--color-text)] transition-colors"
              >
                <X size={16} strokeWidth={1.75} />
              </button>
            </div>

            {aiResponse && (
              <div className="mb-4 px-3 py-2 rounded-lg bg-[var(--color-bg)] border border-[var(--color-border)]">
                <p className="text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)] mb-1">
                  AI Response
                </p>
                <p className="text-[11px] text-[var(--color-text-muted)] leading-relaxed line-clamp-3">
                  {aiResponse.slice(0, 400)}
                  {aiResponse.length > 400 && '…'}
                </p>
              </div>
            )}

            <form onSubmit={handleSubmit} className="space-y-3">
              <textarea
                value={comment}
                onChange={(e) => setComment(e.target.value)}
                placeholder="Was this response helpful? Accurate? Off the mark? Tell us what you think…"
                rows={4}
                maxLength={2000}
                className={cn(
                  'w-full resize-none rounded-xl border border-[var(--color-border)]',
                  'bg-[var(--color-bg)] px-4 py-3 text-sm text-[var(--color-text)]',
                  'placeholder:text-[var(--color-text-muted)] outline-none',
                  'focus:border-[var(--color-accent)]/50 transition-colors duration-150',
                )}
              />
              <div className="flex items-center justify-between">
                <span className="text-[10px] text-[var(--color-text-muted)]">
                  {comment.length}/2000
                </span>
                <button
                  type="submit"
                  disabled={!comment.trim() || status === 'sending' || status === 'done'}
                  className={cn(
                    'inline-flex items-center gap-1.5 rounded-full px-4 py-2 text-xs font-medium transition-all duration-150',
                    status === 'done'
                      ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/20'
                      : 'bg-[var(--color-accent)] text-[var(--color-accent-contrast)] hover:opacity-90 disabled:opacity-40 disabled:cursor-not-allowed',
                  )}
                >
                  {status === 'done' ? (
                    <><Check size={11} strokeWidth={2} /> Sent</>
                  ) : status === 'sending' ? (
                    <span className="h-3 w-3 border border-current border-t-transparent rounded-full animate-spin" />
                  ) : (
                    <><Send size={11} strokeWidth={1.75} /> Submit</>
                  )}
                </button>
              </div>
              {status === 'error' && (
                <p className="text-[11px] text-red-400">Failed to submit — please try again.</p>
              )}
            </form>
          </div>
        </div>
      )}
    </>
  )
}
