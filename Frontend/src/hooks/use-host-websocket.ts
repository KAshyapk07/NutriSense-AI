/**
 * useHostWebSocket — WebSocket connection for the PC (host) side of the
 * AI Chef Kitchen Remote.
 *
 * Connects to `/ws/chef-voice/{sessionId}` as the "host" role.
 * Receives voice intents, phone button actions, and chat messages relayed
 * by the FastAPI backend.  Pushes CookingSessionState updates to the phone
 * through the same channel.
 *
 * This replaces the legacy PeerJS WebRTC data channel on the host side.
 * Standard WebSocket only — fully MS Store sandbox compatible.
 */

import { useState, useEffect, useRef, useCallback, useMemo } from 'react'
import type { CookingSessionState, ChefIntentResponse } from '@/lib/types'

// ── Types ──────────────────────────────────────────────────────────

export type HostConnectionStatus = 'connecting' | 'connected' | 'disconnected' | 'error'

export interface UseHostWebSocketReturn {
  /** Stable session ID for the QR code URL. */
  sessionId: string
  /** Whether the WebSocket to the backend is open. */
  wsConnected: boolean
  /** True once a phone has joined this session. */
  phoneConnected: boolean
  /** Live transcript text from the phone's voice (clears after 3 s). */
  transcript: string
  /** Human-readable error, or null. */
  errorMsg: string | null
  /** Push updated CookingSessionState to the phone. */
  sendState: (state: CookingSessionState) => void
  /** Send a chat reply message to the phone. */
  sendChatReply: (msg: { role: 'user' | 'assistant'; content: string }) => void
  /** Register callback for voice intents processed by the backend. */
  onVoiceIntent: (cb: (intent: ChefIntentResponse & { raw_text?: string }) => void) => void
  /** Register callback for structured button actions from the phone. */
  onPhoneAction: (cb: (action: string, payload?: Record<string, unknown>) => void) => void
  /** Register callback for chat questions from the phone. */
  onPhoneChat: (cb: (text: string) => void) => void
}

// ── Constants ──────────────────────────────────────────────────────

const MAX_RECONNECT_ATTEMPTS = 8
const BASE_RECONNECT_DELAY_MS = 500
const MAX_RECONNECT_DELAY_MS = 15_000

/** Generate a short, URL-safe random session ID. */
function generateSessionId(): string {
  const chars = 'abcdefghijklmnopqrstuvwxyz0123456789'
  let id = 'chef-'
  for (let i = 0; i < 10; i++) {
    id += chars[Math.floor(Math.random() * chars.length)]
  }
  return id
}

// ── Hook ───────────────────────────────────────────────────────────

export function useHostWebSocket(): UseHostWebSocketReturn {
  // Session ID is generated once and stays stable across reconnects.
  const [sessionId] = useState(generateSessionId)

  const [wsConnected, setWsConnected] = useState(false)
  const [phoneConnected, setPhoneConnected] = useState(false)
  const [transcript, setTranscript] = useState('')
  const [errorMsg, setErrorMsg] = useState<string | null>(null)

  const wsRef = useRef<WebSocket | null>(null)
  const reconnectAttemptRef = useRef(0)
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const closedIntentionallyRef = useRef(false)
  const hasConnectedRef = useRef(false)

  // Callback refs
  const voiceIntentCbRef = useRef<((intent: ChefIntentResponse & { raw_text?: string }) => void) | null>(null)
  const phoneActionCbRef = useRef<((action: string, payload?: Record<string, unknown>) => void) | null>(null)
  const phoneChatCbRef = useRef<((text: string) => void) | null>(null)

  // Latest state — sent with init handshake on reconnect
  const latestStateRef = useRef<CookingSessionState | null>(null)

  // ── Derived WebSocket URL ───────────────────────────────────────

  const wsUrl = useMemo(() => {
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    return `${proto}//${window.location.host}/ws/chef-voice/${encodeURIComponent(sessionId)}`
  }, [sessionId])

  // ── Message handling ────────────────────────────────────────────

  const handleMessage = useCallback((data: Record<string, unknown>) => {
    const type = data.type as string

    switch (type) {
      case 'connected':
        break

      case 'peer-joined':
        if (data.role === 'phone') setPhoneConnected(true)
        break

      case 'peer-left':
        if (data.role === 'phone') setPhoneConnected(false)
        break

      case 'voice-intent':
        voiceIntentCbRef.current?.(data as unknown as ChefIntentResponse & { raw_text?: string })
        break

      case 'transcript':
        if (typeof data.text === 'string') {
          setTranscript(data.text)
          if (data.final) {
            setTimeout(() => setTranscript(''), 3000)
          }
        }
        break

      case 'action':
        phoneActionCbRef.current?.(
          data.action as string,
          data as Record<string, unknown>,
        )
        break

      case 'chat':
        if (typeof data.text === 'string') {
          phoneChatCbRef.current?.(data.text)
        }
        break

      case 'error':
        setErrorMsg(typeof data.message === 'string' ? data.message : 'Server error')
        break
    }
  }, [])

  // ── Send helpers ────────────────────────────────────────────────

  const sendJson = useCallback((msg: Record<string, unknown>) => {
    const ws = wsRef.current
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(msg))
    }
  }, [])

  const sendState = useCallback(
    (state: CookingSessionState) => {
      latestStateRef.current = state
      sendJson({ type: 'state', payload: state })
    },
    [sendJson],
  )

  const sendChatReply = useCallback(
    (msg: { role: 'user' | 'assistant'; content: string }) => {
      sendJson({ type: 'chat-reply', ...msg })
    },
    [sendJson],
  )

  const onVoiceIntent = useCallback(
    (cb: (intent: ChefIntentResponse & { raw_text?: string }) => void) => {
      voiceIntentCbRef.current = cb
    },
    [],
  )

  const onPhoneAction = useCallback(
    (cb: (action: string, payload?: Record<string, unknown>) => void) => {
      phoneActionCbRef.current = cb
    },
    [],
  )

  const onPhoneChat = useCallback((cb: (text: string) => void) => {
    phoneChatCbRef.current = cb
  }, [])

  // ── Connection lifecycle ────────────────────────────────────────

  const connect = useCallback(() => {
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

    const ws = new WebSocket(wsUrl)
    wsRef.current = ws

    ws.onopen = () => {
      setWsConnected(true)
      setErrorMsg(null)
      reconnectAttemptRef.current = 0
      hasConnectedRef.current = true

      // Init handshake — identify as host and push cached state
      const init: Record<string, unknown> = { type: 'init', role: 'host' }
      if (latestStateRef.current) {
        init.state = latestStateRef.current
      }
      ws.send(JSON.stringify(init))
    }

    ws.onmessage = (event: MessageEvent) => {
      if (typeof event.data !== 'string') return
      try {
        handleMessage(JSON.parse(event.data))
      } catch {
        // Non-JSON — ignore
      }
    }

    ws.onclose = () => {
      setWsConnected(false)
      if (!closedIntentionallyRef.current) {
        scheduleReconnect()
      }
    }

    ws.onerror = () => {
      if (!hasConnectedRef.current) {
        setErrorMsg('Could not connect to the voice server.')
      }
    }
  }, [wsUrl, handleMessage])

  /**
   * Exponential-backoff reconnect: 500 → 1 000 → 2 000 → … → 15 000 ms cap.
   */
  const scheduleReconnect = useCallback(() => {
    if (closedIntentionallyRef.current) return
    if (reconnectAttemptRef.current >= MAX_RECONNECT_ATTEMPTS) {
      setWsConnected(false)
      setErrorMsg('Lost connection to the server. Please refresh the page.')
      return
    }

    const attempt = reconnectAttemptRef.current++
    const delay = Math.min(BASE_RECONNECT_DELAY_MS * 2 ** attempt, MAX_RECONNECT_DELAY_MS)

    reconnectTimerRef.current = setTimeout(() => {
      if (!closedIntentionallyRef.current) connect()
    }, delay)
  }, [connect])

  // ── Mount/unmount ───────────────────────────────────────────────

  useEffect(() => {
    closedIntentionallyRef.current = false
    connect()

    return () => {
      closedIntentionallyRef.current = true

      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current)
        reconnectTimerRef.current = null
      }

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
    }
  }, [connect])

  return {
    sessionId,
    wsConnected,
    phoneConnected,
    transcript,
    errorMsg,
    sendState,
    sendChatReply,
    onVoiceIntent,
    onPhoneAction,
    onPhoneChat,
  }
}
