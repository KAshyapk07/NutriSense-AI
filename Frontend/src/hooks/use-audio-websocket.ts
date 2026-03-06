/**
 * useAudioWebSocket — WebSocket-based audio streaming hook for the Kitchen Remote.
 *
 * Replaces the legacy PeerJS + Web Speech API architecture with a direct
 * WebSocket connection to the FastAPI backend.  Audio captured via the
 * MediaRecorder API (WebM/Opus, 250 ms timeslice) is streamed as binary
 * frames to the server, which handles STT, intent parsing, and state relay.
 *
 * This hook is MS Store compatible: it uses only `getUserMedia` for mic
 * access (a DeviceCapability in the AppxManifest) and standard WebSockets
 * — no WebRTC, no third-party broker, no browser-only Speech API tokens.
 *
 * Features:
 *  - Persistent WebSocket with exponential-backoff auto-reconnect.
 *  - MediaRecorder lifecycle management (start/stop/permission handling).
 *  - Typed JSON message handling for state, intents, transcripts, and chat.
 *  - Clean teardown on unmount (stops media tracks, closes socket).
 */

import {
  useState,
  useEffect,
  useRef,
  useCallback,
  useMemo,
} from 'react'
import type { CookingSessionState, ChefIntentResponse } from '@/lib/types'

// ── Types ──────────────────────────────────────────────────────────

export type ConnectionStatus = 'connecting' | 'connected' | 'disconnected' | 'error'

export interface ChatMsg {
  role: 'user' | 'assistant'
  content: string
}

/** Incoming WebSocket JSON message from the backend. */
interface WsMessage {
  type: string
  [key: string]: unknown
}

export interface UseAudioWebSocketOptions {
  /** Session ID from the QR code URL (maps phone to PC session). */
  sessionId: string | null
  /** Connection role — "phone" for kitchen remote, "host" for PC. */
  role?: 'phone' | 'host'
  /** MediaRecorder timeslice in ms (default 250). Smaller = lower latency. */
  timeslice?: number
  /** Initial CookingSessionState to send with the init handshake. */
  initialState?: CookingSessionState | null
}

export interface UseAudioWebSocketReturn {
  /** Current WebSocket connection status. */
  connectionStatus: ConnectionStatus
  /** Latest CookingSessionState received from the host PC. */
  sessionState: CookingSessionState | null
  /** Whether the microphone is actively recording and streaming. */
  listening: boolean
  /** Live transcript text from the backend STT engine. */
  transcript: string
  /** Chat messages accumulated during this session. */
  chatMessages: ChatMsg[]
  /** Whether we're waiting for a chat response from the host. */
  chatLoading: boolean
  /** Human-readable error message, or null. */
  errorMsg: string | null
  /** Start capturing audio from the mic and streaming to the backend. */
  startRecording: () => void
  /** Stop recording and flush any pending audio to the backend. */
  stopRecording: () => void
  /** Send a structured action (next, prev, timer-start, etc.) via WebSocket. */
  sendAction: (action: string, extra?: Record<string, unknown>) => void
  /** Send a chat question to the host PC via the backend relay. */
  sendChat: (text: string) => void
  /** Register a callback for voice intent responses from the backend. */
  onIntent: (cb: (intent: ChefIntentResponse) => void) => void
}

// ── Constants ──────────────────────────────────────────────────────

/** Maximum reconnect attempts before giving up. */
const MAX_RECONNECT_ATTEMPTS = 8
/** Base delay for exponential backoff (doubles each attempt, capped). */
const BASE_RECONNECT_DELAY_MS = 500
/** Maximum reconnect delay (cap for exponential backoff). */
const MAX_RECONNECT_DELAY_MS = 15_000

// ── Hook ───────────────────────────────────────────────────────────

export function useAudioWebSocket({
  sessionId,
  role = 'phone',
  timeslice = 250,
  initialState = null,
}: UseAudioWebSocketOptions): UseAudioWebSocketReturn {
  // ── React state exposed to the consuming component ──────────────
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>('connecting')
  const [sessionState, setSessionState] = useState<CookingSessionState | null>(null)
  const [listening, setListening] = useState(false)
  const [transcript, setTranscript] = useState('')
  const [chatMessages, setChatMessages] = useState<ChatMsg[]>([])
  const [chatLoading, setChatLoading] = useState(false)
  const [errorMsg, setErrorMsg] = useState<string | null>(null)

  // ── Refs for mutable state not triggering re-renders ────────────
  const wsRef = useRef<WebSocket | null>(null)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const reconnectAttemptRef = useRef(0)
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const intentCallbackRef = useRef<((intent: ChefIntentResponse) => void) | null>(null)
  /** Prevents reconnect loops after intentional close (unmount). */
  const closedIntentionallyRef = useRef(false)
  /** Tracks whether we ever successfully connected (for reconnect logic). */
  const hasConnectedRef = useRef(false)

  // ── Derived WebSocket URL ───────────────────────────────────────
  const wsUrl = useMemo(() => {
    if (!sessionId) return null
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    return `${proto}//${window.location.host}/ws/chef-voice/${encodeURIComponent(sessionId)}`
  }, [sessionId])

  // ── Message handling ────────────────────────────────────────────

  const handleMessage = useCallback((msg: WsMessage) => {
    switch (msg.type) {
      case 'connected':
        // Handshake acknowledged by the server
        break

      case 'state':
        // CookingSessionState push from the host PC (relayed by backend)
        if (msg.payload && typeof msg.payload === 'object') {
          setSessionState(msg.payload as CookingSessionState)
        }
        break

      case 'transcript':
        // Live transcript from backend STT
        if (typeof msg.text === 'string') {
          setTranscript(msg.text)
          // Clear transcript after a short display window when final
          if (msg.final) {
            setTimeout(() => setTranscript(''), 3000)
          }
        }
        break

      case 'intent': {
        // ChefIntentResponse from voice command processing
        const intent = msg as unknown as ChefIntentResponse & { type: string }
        intentCallbackRef.current?.(intent)
        break
      }

      case 'chat-reply':
        // Chat answer relayed from the host
        if (typeof msg.content === 'string') {
          setChatMessages((prev) => [
            ...prev,
            { role: (msg.role as 'user' | 'assistant') ?? 'assistant', content: msg.content as string },
          ])
          if (msg.role === 'assistant') setChatLoading(false)
        }
        break

      case 'peer-joined':
        // The host PC connected to the session
        break

      case 'peer-left':
        // The host PC disconnected (session may have ended)
        if (msg.role === 'host') {
          setSessionState(null)
        }
        break

      case 'error':
        setErrorMsg(typeof msg.message === 'string' ? msg.message : 'Unknown server error')
        break
    }
  }, [])

  // ── WebSocket send helpers ──────────────────────────────────────

  const sendJson = useCallback((data: Record<string, unknown>) => {
    const ws = wsRef.current
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(data))
    }
  }, [])

  const sendAction = useCallback(
    (action: string, extra?: Record<string, unknown>) => {
      sendJson({ type: 'action', action, ...extra })
    },
    [sendJson],
  )

  const sendChat = useCallback(
    (text: string) => {
      const trimmed = text.trim()
      if (!trimmed) return
      setChatMessages((prev) => [...prev, { role: 'user', content: trimmed }])
      setChatLoading(true)
      sendJson({ type: 'chat', text: trimmed })
    },
    [sendJson],
  )

  const onIntent = useCallback((cb: (intent: ChefIntentResponse) => void) => {
    intentCallbackRef.current = cb
  }, [])

  // ── WebSocket connection lifecycle ──────────────────────────────

  const connect = useCallback(() => {
    if (!wsUrl) {
      setConnectionStatus('error')
      setErrorMsg('No session ID — scan the QR code from your PC.')
      return
    }

    // Clean up any previous socket
    if (wsRef.current) {
      wsRef.current.onopen = null
      wsRef.current.onmessage = null
      wsRef.current.onclose = null
      wsRef.current.onerror = null
      if (wsRef.current.readyState < WebSocket.CLOSING) {
        wsRef.current.close()
      }
    }

    setConnectionStatus('connecting')
    const ws = new WebSocket(wsUrl)
    wsRef.current = ws

    ws.binaryType = 'arraybuffer'

    ws.onopen = () => {
      setConnectionStatus('connected')
      setErrorMsg(null)
      reconnectAttemptRef.current = 0
      hasConnectedRef.current = true

      // Send the init handshake so the backend knows our role
      const initPayload: Record<string, unknown> = { type: 'init', role }
      if (initialState) {
        initPayload.state = initialState
      }
      ws.send(JSON.stringify(initPayload))
    }

    ws.onmessage = (event: MessageEvent) => {
      if (typeof event.data !== 'string') return
      try {
        const msg = JSON.parse(event.data) as WsMessage
        handleMessage(msg)
      } catch {
        // Non-JSON message — ignore
      }
    }

    ws.onclose = (event) => {
      // Only attempt reconnect if we didn't intentionally close
      if (!closedIntentionallyRef.current) {
        setConnectionStatus('disconnected')
        scheduleReconnect()
      }
    }

    ws.onerror = () => {
      // onerror always fires before onclose — don't double-set status
      if (!hasConnectedRef.current) {
        setErrorMsg('Could not connect to the server.')
      }
    }
  }, [wsUrl, role, initialState, handleMessage])

  /**
   * Exponential-backoff reconnect scheduler.
   *
   * Delay doubles each attempt: 500 → 1000 → 2000 → 4000 → 8000 → 15000 (cap).
   * Gives up after MAX_RECONNECT_ATTEMPTS.
   */
  const scheduleReconnect = useCallback(() => {
    if (closedIntentionallyRef.current) return
    if (reconnectAttemptRef.current >= MAX_RECONNECT_ATTEMPTS) {
      setConnectionStatus('error')
      setErrorMsg(
        'Lost connection to the server. Please refresh the page or scan the QR code again.',
      )
      return
    }

    const attempt = reconnectAttemptRef.current++
    const delay = Math.min(BASE_RECONNECT_DELAY_MS * 2 ** attempt, MAX_RECONNECT_DELAY_MS)

    reconnectTimerRef.current = setTimeout(() => {
      if (!closedIntentionallyRef.current) {
        connect()
      }
    }, delay)
  }, [connect])

  // ── Connect on mount, disconnect on unmount ─────────────────────

  useEffect(() => {
    closedIntentionallyRef.current = false
    connect()

    return () => {
      closedIntentionallyRef.current = true

      // Cancel pending reconnect
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current)
        reconnectTimerRef.current = null
      }

      // Close WebSocket
      const ws = wsRef.current
      if (ws) {
        ws.onopen = null
        ws.onmessage = null
        ws.onclose = null
        ws.onerror = null
        if (ws.readyState < WebSocket.CLOSING) {
          ws.close(1000, 'Component unmounted')
        }
        wsRef.current = null
      }

      // Stop media tracks
      streamRef.current?.getTracks().forEach((t) => t.stop())
      streamRef.current = null
      mediaRecorderRef.current = null
    }
  }, [connect])

  // ── MediaRecorder lifecycle ─────────────────────────────────────

  /**
   * Start capturing audio from the microphone.
   *
   * Requests mic permission via `getUserMedia`, creates a `MediaRecorder`
   * with WebM/Opus encoding, and streams binary chunks to the WebSocket
   * at the configured `timeslice` interval (default 250 ms).
   *
   * If mic permission is denied, sets an error message and returns
   * gracefully — never throws.
   */
  const startRecording = useCallback(async () => {
    // Don't start if WS isn't connected
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return

    // Already recording
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') return

    try {
      // Request microphone access.  On MS Store, this requires the
      // <DeviceCapability Name="microphone" /> in the AppxManifest.
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          // Optimize for voice: disable noise suppression and echo
          // cancellation if the device supports it (kitchen environment).
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          sampleRate: 16000,
        },
        video: false,
      })
      streamRef.current = stream

      // Prefer WebM/Opus — universally supported, efficient for speech.
      // Fall back to whatever the browser offers if Opus isn't available.
      const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
        ? 'audio/webm;codecs=opus'
        : MediaRecorder.isTypeSupported('audio/webm')
          ? 'audio/webm'
          : ''

      const recorder = new MediaRecorder(stream, {
        ...(mimeType ? { mimeType } : {}),
      })

      recorder.ondataavailable = (event: BlobEvent) => {
        // Stream each chunk as a binary WebSocket frame.
        // The backend's VoiceActivityDetector uses chunk sizes to
        // detect speech boundaries (Opus VBR: silence ≈ 30–150 B,
        // speech ≈ 400–3000 B per 250 ms timeslice).
        if (event.data.size > 0 && wsRef.current?.readyState === WebSocket.OPEN) {
          event.data.arrayBuffer().then((buf) => {
            wsRef.current?.send(buf)
          })
        }
      }

      recorder.onstart = () => {
        setListening(true)
        setErrorMsg(null)
      }

      recorder.onstop = () => {
        setListening(false)
      }

      recorder.onerror = () => {
        setListening(false)
        setErrorMsg('Microphone error — try toggling the mic button.')
      }

      mediaRecorderRef.current = recorder
      recorder.start(timeslice)
    } catch (err) {
      // getUserMedia failures: permission denied, no mic, or secure context required
      const error = err as DOMException
      if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
        setErrorMsg('Microphone access denied. Please allow mic permission and try again.')
      } else if (error.name === 'NotFoundError') {
        setErrorMsg('No microphone found. Please connect a mic and try again.')
      } else {
        setErrorMsg(`Microphone error: ${error.message}`)
      }
      setListening(false)
    }
  }, [timeslice])

  /**
   * Stop the MediaRecorder and notify the backend to flush pending audio.
   *
   * Sends a ``stop-recording`` control message so the backend's VAD
   * force-flushes any accumulated speech buffer for a final transcription.
   */
  const stopRecording = useCallback(() => {
    const recorder = mediaRecorderRef.current
    if (recorder && recorder.state !== 'inactive') {
      recorder.stop()
    }

    // Stop all media tracks to release the microphone
    streamRef.current?.getTracks().forEach((t) => t.stop())
    streamRef.current = null
    mediaRecorderRef.current = null

    // Tell the backend to flush any pending audio
    sendJson({ type: 'stop-recording' })

    setListening(false)
    setTranscript('')
  }, [sendJson])

  return {
    connectionStatus,
    sessionState,
    listening,
    transcript,
    chatMessages,
    chatLoading,
    errorMsg,
    startRecording,
    stopRecording,
    sendAction,
    sendChat,
    onIntent,
  }
}
