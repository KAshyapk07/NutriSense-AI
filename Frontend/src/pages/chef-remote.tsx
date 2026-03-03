/**
 * Chef Remote — Phone-side kitchen voice controller.
 * Route: /chef-remote?peer=<peerId>
 *
 * This page is designed to be opened on a phone by scanning a QR code.
 * It connects to the desktop PC via PeerJS (WebRTC data channel),
 * captures voice commands using the Web Speech API, and displays
 * the current cooking state synced from the PC.
 *
 * The phone does NO AI processing — it is purely a remote mic + display.
 * All intent parsing happens on the PC's local FastAPI + Ollama.
 *
 * Screen Wake Lock keeps the phone screen alive in the kitchen.
 */
import { useState, useEffect, useRef, useCallback } from 'react'
import Peer, { type DataConnection } from 'peerjs'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Mic,
  MicOff,
  ChefHat,
  Wifi,
  WifiOff,
  Timer,
  CheckCircle2,
  Circle,
  Wrench,
  Lightbulb,
  ArrowRight,
  ArrowLeft,
} from 'lucide-react'
import { cn } from '@/lib/utils'
import type { CookingSessionState } from '@/lib/types'

// ── Helpers ────────────────────────────────────────────────────────

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60)
  const s = seconds % 60
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
}

/**
 * Get the peer ID from the URL query string.
 */
function getPeerIdFromUrl(): string | null {
  const params = new URLSearchParams(window.location.search)
  return params.get('peer')
}

// ── Web Speech API types ────────────────────────────────────────

interface SpeechRecognitionEvent extends Event {
  results: SpeechRecognitionResultList
  resultIndex: number
}

interface SpeechRecognitionErrorEvent extends Event {
  error: string
  message?: string
}

interface SpeechRecognitionInstance extends EventTarget {
  continuous: boolean
  interimResults: boolean
  lang: string
  start(): void
  stop(): void
  abort(): void
  onresult: ((event: SpeechRecognitionEvent) => void) | null
  onerror: ((event: SpeechRecognitionErrorEvent) => void) | null
  onend: (() => void) | null
  onstart: (() => void) | null
}

declare global {
  interface Window {
    SpeechRecognition?: new () => SpeechRecognitionInstance
    webkitSpeechRecognition?: new () => SpeechRecognitionInstance
  }
}

// ── Main Component ──────────────────────────────────────────────

export default function ChefRemotePage() {
  const [connectionStatus, setConnectionStatus] = useState<
    'connecting' | 'connected' | 'disconnected' | 'error'
  >('connecting')
  const [errorMsg, setErrorMsg] = useState<string | null>(null)
  const [sessionState, setSessionState] =
    useState<CookingSessionState | null>(null)
  const [listening, setListening] = useState(false)
  const [transcript, setTranscript] = useState('')
  const [lastSent, setLastSent] = useState<string | null>(null)
  const [wakeLockActive, setWakeLockActive] = useState(false)

  const connRef = useRef<DataConnection | null>(null)
  const recognitionRef = useRef<SpeechRecognitionInstance | null>(null)
  const wakeLockRef = useRef<WakeLockSentinel | null>(null)

  // ── Connect to PC via PeerJS ───────────────────────────────────

  useEffect(() => {
    const hostPeerId = getPeerIdFromUrl()
    if (!hostPeerId) {
      setConnectionStatus('error')
      setErrorMsg('No peer ID in URL. Please scan the QR code again.')
      return
    }

    const peer = new Peer({ debug: 0 })

    peer.on('open', () => {
      const conn = peer.connect(hostPeerId, { reliable: true })
      connRef.current = conn

      conn.on('open', () => {
        setConnectionStatus('connected')
        setErrorMsg(null)
      })

      conn.on('data', (data) => {
        if (
          typeof data === 'object' &&
          data !== null &&
          'type' in data &&
          (data as Record<string, unknown>).type === 'state'
        ) {
          setSessionState(
            (data as { type: string; payload: CookingSessionState }).payload,
          )
        }
      })

      conn.on('close', () => {
        setConnectionStatus('disconnected')
      })

      conn.on('error', (err) => {
        setConnectionStatus('error')
        setErrorMsg(err.message)
      })
    })

    peer.on('error', (err) => {
      setConnectionStatus('error')
      setErrorMsg(`Connection failed: ${err.message}`)
    })

    return () => {
      peer.destroy()
    }
  }, [])

  // ── Screen Wake Lock ───────────────────────────────────────────

  useEffect(() => {
    let sentinel: WakeLockSentinel | null = null

    async function requestWakeLock() {
      try {
        if ('wakeLock' in navigator) {
          sentinel = await navigator.wakeLock.request('screen')
          wakeLockRef.current = sentinel
          setWakeLockActive(true)

          sentinel.addEventListener('release', () => {
            setWakeLockActive(false)
          })
        }
      } catch {
        // Wake Lock not supported or denied — degrade gracefully
        setWakeLockActive(false)
      }
    }

    if (connectionStatus === 'connected') {
      requestWakeLock()
    }

    // Re-acquire on visibility change (e.g. phone screen turned on)
    function handleVisibilityChange() {
      if (document.visibilityState === 'visible' && connectionStatus === 'connected') {
        requestWakeLock()
      }
    }

    document.addEventListener('visibilitychange', handleVisibilityChange)

    return () => {
      document.removeEventListener('visibilitychange', handleVisibilityChange)
      sentinel?.release()
    }
  }, [connectionStatus])

  // ── Voice Recognition (Web Speech API) ─────────────────────────

  const sendVoiceText = useCallback(
    (text: string) => {
      const conn = connRef.current
      if (conn?.open && text.trim()) {
        conn.send({ type: 'voice', text: text.trim() })
        setLastSent(text.trim())
      }
    },
    [],
  )

  const startListening = useCallback(() => {
    const SpeechRecognition =
      window.SpeechRecognition || window.webkitSpeechRecognition
    if (!SpeechRecognition) {
      setErrorMsg('Speech recognition not supported in this browser.')
      return
    }

    const recognition = new SpeechRecognition()
    recognition.continuous = true
    recognition.interimResults = true
    recognition.lang = 'en-US'

    recognition.onstart = () => {
      setListening(true)
    }

    recognition.onresult = (event: SpeechRecognitionEvent) => {
      let finalTranscript = ''
      let interimTranscript = ''

      for (let i = event.resultIndex; i < event.results.length; i++) {
        const result = event.results[i]
        if (result.isFinal) {
          finalTranscript += result[0].transcript
        } else {
          interimTranscript += result[0].transcript
        }
      }

      setTranscript(interimTranscript || finalTranscript)

      if (finalTranscript.trim()) {
        sendVoiceText(finalTranscript)
        setTranscript('')
      }
    }

    recognition.onerror = (event: SpeechRecognitionErrorEvent) => {
      if (event.error !== 'no-speech' && event.error !== 'aborted') {
        console.error('Speech recognition error:', event.error)
      }
    }

    recognition.onend = () => {
      setListening(false)
      // Auto-restart if still connected
      if (connRef.current?.open) {
        try {
          recognition.start()
        } catch {
          // Already started or destroyed
        }
      }
    }

    recognitionRef.current = recognition
    recognition.start()
  }, [sendVoiceText])

  const stopListening = useCallback(() => {
    recognitionRef.current?.stop()
    recognitionRef.current = null
    setListening(false)
    setTranscript('')
  }, [])

  // Clean up on unmount
  useEffect(() => {
    return () => {
      recognitionRef.current?.abort()
    }
  }, [])

  // ── Quick action buttons (send text commands directly) ─────────

  const sendQuickCommand = useCallback(
    (command: string) => {
      sendVoiceText(command)
    },
    [sendVoiceText],
  )

  // ── Render ─────────────────────────────────────────────────────

  // Connecting / Error state
  if (connectionStatus !== 'connected') {
    return (
      <div className="min-h-screen bg-[#0a0a0a] text-white flex flex-col items-center justify-center p-6">
        <div className="flex flex-col items-center gap-6 text-center max-w-xs">
          <div className="p-4 rounded-full bg-white/5">
            {connectionStatus === 'error' ? (
              <WifiOff size={36} className="text-red-400" />
            ) : (
              <div className="w-10 h-10 rounded-full border-3 border-white/20 border-t-white/80 animate-spin" />
            )}
          </div>

          <div>
            <h1 className="text-xl font-bold mb-1">NutriSense Kitchen Remote</h1>
            <p className="text-sm text-white/50">
              {connectionStatus === 'connecting' && 'Connecting to your PC...'}
              {connectionStatus === 'disconnected' && 'Connection lost. Close and scan the QR code again.'}
              {connectionStatus === 'error' && (errorMsg ?? 'Connection failed.')}
            </p>
          </div>

          {connectionStatus === 'error' && (
            <button
              onClick={() => window.location.reload()}
              className="px-5 py-2.5 rounded-xl bg-white/10 text-sm font-medium hover:bg-white/15 transition-colors"
            >
              Try Again
            </button>
          )}
        </div>
      </div>
    )
  }

  // Connected — show cooking remote
  return (
    <div className="min-h-screen bg-[#0a0a0a] text-white flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-white/10">
        <div className="flex items-center gap-2">
          <ChefHat size={18} className="text-orange-400" />
          <span className="text-sm font-semibold">Kitchen Remote</span>
        </div>
        <div className="flex items-center gap-2 text-xs text-green-400">
          <Wifi size={13} />
          Connected
          {wakeLockActive && (
            <span className="text-white/30 ml-1">screen locked</span>
          )}
        </div>
      </div>

      {/* Main content */}
      <div className="flex-1 flex flex-col p-4 gap-4 overflow-y-auto">
        {sessionState ? (
          <>
            {/* Recipe name + step indicator */}
            <div className="flex items-center justify-between">
              <h2 className="text-base font-bold capitalize truncate">
                {sessionState.recipe_name}
              </h2>
              <span className="text-sm text-white/50 tabular-nums flex-shrink-0">
                Step{' '}
                <span className="text-orange-400 font-bold">
                  {sessionState.current_step}
                </span>
                {' / '}
                {sessionState.total_steps}
              </span>
            </div>

            {/* Step progress dots */}
            <div className="flex gap-1 items-center">
              {sessionState.steps_overview.map((s, i) => {
                const isCurrent = i + 1 === sessionState.current_step
                const isCompleted = s.completed === 'true'
                return (
                  <div
                    key={s.id}
                    className={cn(
                      'h-1.5 rounded-full transition-all',
                      isCurrent
                        ? 'w-6 bg-orange-400'
                        : isCompleted
                          ? 'w-3 bg-green-500'
                          : 'w-3 bg-white/15',
                    )}
                  />
                )
              })}
            </div>

            {/* Current step card */}
            <AnimatePresence mode="wait">
              <motion.div
                key={sessionState.current_step}
                initial={{ opacity: 0, x: 30 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -30 }}
                className="rounded-2xl bg-white/5 border border-white/10 p-5"
              >
                {sessionState.current_tool && (
                  <div className="flex items-center gap-1.5 mb-2">
                    <Wrench size={12} className="text-white/40" />
                    <span className="text-[11px] text-white/40 capitalize">
                      {sessionState.current_tool}
                    </span>
                  </div>
                )}

                <p className="text-lg font-semibold leading-relaxed">
                  {sessionState.current_action}
                </p>

                {sessionState.current_tip && (
                  <div className="mt-4 flex items-start gap-2 p-3 rounded-xl bg-amber-500/10 border border-amber-500/20">
                    <Lightbulb size={14} className="text-amber-400 flex-shrink-0 mt-0.5" />
                    <p className="text-sm text-amber-200/80 leading-relaxed">
                      {sessionState.current_tip}
                    </p>
                  </div>
                )}

                {/* Timer */}
                {sessionState.timer_total != null && sessionState.timer_left != null && (
                  <div className="mt-5 flex items-center justify-center gap-4">
                    <div className="relative w-20 h-20">
                      <svg className="w-full h-full -rotate-90" viewBox="0 0 100 100">
                        <circle
                          cx="50" cy="50" r="42"
                          fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth="5"
                        />
                        <circle
                          cx="50" cy="50" r="42"
                          fill="none"
                          stroke={
                            sessionState.timer_left === 0
                              ? '#22c55e'
                              : sessionState.timer_left <= 30
                                ? '#ef4444'
                                : '#f97316'
                          }
                          strokeWidth="5"
                          strokeLinecap="round"
                          strokeDasharray={`${2 * Math.PI * 42}`}
                          strokeDashoffset={`${
                            2 * Math.PI * 42 * (1 - sessionState.timer_left / sessionState.timer_total)
                          }`}
                          style={{ transition: 'stroke-dashoffset 0.9s linear' }}
                        />
                      </svg>
                      <div className="absolute inset-0 flex items-center justify-center">
                        <span
                          className={cn(
                            'text-lg font-bold tabular-nums',
                            sessionState.timer_left === 0 && 'text-green-400',
                            sessionState.timer_left <= 30 && sessionState.timer_left > 0 && 'text-red-400',
                          )}
                        >
                          {sessionState.timer_left === 0
                            ? 'Done'
                            : formatTime(sessionState.timer_left)}
                        </span>
                      </div>
                    </div>
                    <div className="text-xs text-white/40">
                      {sessionState.timer_running ? (
                        <span className="text-orange-400">Running</span>
                      ) : sessionState.timer_left === 0 ? (
                        <span className="text-green-400">Complete</span>
                      ) : (
                        'Paused'
                      )}
                    </div>
                  </div>
                )}
              </motion.div>
            </AnimatePresence>

            {/* Quick action buttons */}
            <div className="grid grid-cols-3 gap-2">
              <button
                onClick={() => sendQuickCommand('previous step')}
                className="flex items-center justify-center gap-1.5 py-3 rounded-xl bg-white/5 border border-white/10 text-sm font-medium active:bg-white/10 transition-colors"
              >
                <ArrowLeft size={14} /> Prev
              </button>
              <button
                onClick={() => sendQuickCommand('done with this step')}
                className="flex items-center justify-center gap-1.5 py-3 rounded-xl bg-orange-500/20 border border-orange-500/30 text-orange-300 text-sm font-bold active:bg-orange-500/30 transition-colors"
              >
                <CheckCircle2 size={14} /> Done
              </button>
              <button
                onClick={() => sendQuickCommand('next step')}
                className="flex items-center justify-center gap-1.5 py-3 rounded-xl bg-white/5 border border-white/10 text-sm font-medium active:bg-white/10 transition-colors"
              >
                Next <ArrowRight size={14} />
              </button>
            </div>

            {/* Timer controls */}
            {sessionState.timer_total != null && (
              <div className="grid grid-cols-3 gap-2">
                <button
                  onClick={() => sendQuickCommand('start timer')}
                  className="flex items-center justify-center gap-1 py-2.5 rounded-xl bg-white/5 border border-white/10 text-xs font-medium active:bg-white/10 transition-colors"
                >
                  <Timer size={12} /> Start
                </button>
                <button
                  onClick={() => sendQuickCommand('pause timer')}
                  className="flex items-center justify-center gap-1 py-2.5 rounded-xl bg-white/5 border border-white/10 text-xs font-medium active:bg-white/10 transition-colors"
                >
                  Pause
                </button>
                <button
                  onClick={() => sendQuickCommand('reset timer')}
                  className="flex items-center justify-center gap-1 py-2.5 rounded-xl bg-white/5 border border-white/10 text-xs font-medium active:bg-white/10 transition-colors"
                >
                  Reset
                </button>
              </div>
            )}
          </>
        ) : (
          <div className="flex-1 flex flex-col items-center justify-center gap-3 text-center">
            <div className="w-8 h-8 rounded-full border-2 border-white/20 border-t-orange-400 animate-spin" />
            <p className="text-sm text-white/50">
              Waiting for cooking session from PC...
            </p>
          </div>
        )}

        {/* Last sent voice command */}
        {lastSent && (
          <div className="text-xs text-white/30 text-center">
            Last command: &quot;{lastSent}&quot;
          </div>
        )}
      </div>

      {/* Voice control footer */}
      <div className="border-t border-white/10 p-4 pb-8 bg-[#0a0a0a]">
        {/* Live transcript */}
        {transcript && (
          <div className="text-sm text-orange-300/70 text-center mb-3 italic">
            {transcript}...
          </div>
        )}

        <button
          onClick={listening ? stopListening : startListening}
          className={cn(
            'w-full flex items-center justify-center gap-3 py-4 rounded-2xl font-semibold text-base transition-all active:scale-[0.98]',
            listening
              ? 'bg-red-500/20 border-2 border-red-500/50 text-red-300'
              : 'bg-orange-500/20 border-2 border-orange-500/40 text-orange-300',
          )}
        >
          {listening ? (
            <>
              <div className="relative">
                <MicOff size={22} />
                <span className="absolute -top-1 -right-1 w-2.5 h-2.5 rounded-full bg-red-500 animate-pulse" />
              </div>
              Tap to stop listening
            </>
          ) : (
            <>
              <Mic size={22} />
              Tap to start voice control
            </>
          )}
        </button>

        <p className="text-[10px] text-white/25 text-center mt-2">
          Say: &quot;next step&quot;, &quot;done&quot;, &quot;start timer&quot;,
          &quot;repeat that&quot;, or ask a question
        </p>
      </div>
    </div>
  )
}
