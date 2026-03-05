/**
 * usePeerConnection — WebRTC P2P hook via PeerJS.
 *
 * The PC (host) initializes a Peer, gets a unique ID, and waits for the
 * phone (remote) to connect over a DataChannel.  Once connected, the
 * PeerJS broker steps out — all data flows directly P2P.
 *
 * Usage (PC side / host):
 *   const { peerId, connected, sendState, onVoiceText, destroy } = usePeerConnection()
 *
 * The phone side uses PeerJS in chef-remote.tsx directly (lightweight).
 */

import { useCallback, useEffect, useRef, useState } from 'react'
import Peer, { type DataConnection } from 'peerjs'
import type { CookingSessionState } from '@/lib/types'

export interface PeerHookReturn {
  /** Unique peer ID assigned by the broker (shown in QR code). */
  peerId: string | null
  /** True once a phone is connected over WebRTC. */
  connected: boolean
  /** Push updated cooking state to the phone. */
  sendState: (state: CookingSessionState) => void
  /** Register a callback for incoming voice text from the phone. */
  onVoiceText: (cb: (text: string) => void) => void
  /** Tear down the peer connection and clean up. */
  destroy: () => void
  /** Connection error message, if any. */
  error: string | null
}

/**
 * Generates a short, URL-safe random suffix for the peer ID.
 */
function randomSuffix(len = 8): string {
  const chars = 'abcdefghijklmnopqrstuvwxyz0123456789'
  let out = ''
  for (let i = 0; i < len; i++) {
    out += chars[Math.floor(Math.random() * chars.length)]
  }
  return out
}

/**
 * ICE servers for WebRTC — shared between PC host and phone remote.
 * STUN discovers the public IP; TURN relays traffic when direct P2P fails
 * (symmetric NAT, carrier-grade NAT on mobile, corporate firewalls, etc.).
 *
 * The TURN credentials below are from the free metered.ca OpenRelay service.
 * Replace with your own TURN provider for production.
 */
export const ICE_SERVERS: RTCIceServer[] = [
  { urls: 'stun:stun.l.google.com:19302' },
  { urls: 'stun:stun1.l.google.com:19302' },
  {
    urls: 'turn:a.relay.metered.ca:80',
    username: 'e8dd65b92f0bbc2da60a5a95',
    credential: '2jVLShOkmCpuCAHf',
  },
  {
    urls: 'turn:a.relay.metered.ca:443',
    username: 'e8dd65b92f0bbc2da60a5a95',
    credential: '2jVLShOkmCpuCAHf',
  },
  {
    urls: 'turn:a.relay.metered.ca:443?transport=tcp',
    username: 'e8dd65b92f0bbc2da60a5a95',
    credential: '2jVLShOkmCpuCAHf',
  },
]

/**
 * PeerJS constructor options that point to the self-hosted signaling
 * server running on the same FastAPI backend (avoids the unreliable
 * public 0.peerjs.com broker).  Both the PC and phone use this so
 * signaling goes through the same origin / ngrok tunnel.
 */
export function getSelfHostedPeerConfig() {
  const loc = window.location
  return {
    host: loc.hostname,
    port: loc.port ? Number(loc.port) : loc.protocol === 'https:' ? 443 : 80,
    secure: loc.protocol === 'https:',
    path: '/',
    key: 'peerjs',
  }
}

export function usePeerConnection(): PeerHookReturn {
  const [peerId, setPeerId] = useState<string | null>(null)
  const [connected, setConnected] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const peerRef = useRef<Peer | null>(null)
  const connRef = useRef<DataConnection | null>(null)
  const voiceCallbackRef = useRef<((text: string) => void) | null>(null)
  // Phone's peer ID learned from the WebSocket relay handshake
  const relayPeerIdRef = useRef<string | null>(null)

  // Initialize peer on mount
  useEffect(() => {
    const id = `nutri-chef-${randomSuffix()}`
    const peer = new Peer(id, {
      ...getSelfHostedPeerConfig(),
      debug: 1, // basic logging — set to 0 for silent production
      config: { iceServers: ICE_SERVERS },
    })

    peerRef.current = peer

    peer.on('open', (assignedId) => {
      setPeerId(assignedId)
      setError(null)

      // ── WebSocket relay fallback ──────────────────────────────────
      // Listen for RELAY messages on the signaling socket.  The signaling
      // server already forwards any message with a `dst`, so type "RELAY"
      // works out-of-the-box.  This gives us a guaranteed data channel
      // even when WebRTC P2P fails (mobile carrier NAT, expired TURN, etc.).
      peer.socket.on('message', (msg: Record<string, unknown>) => {
        if (msg.type !== 'RELAY' || typeof msg.payload !== 'object' || !msg.payload) return
        const payload = msg.payload as Record<string, unknown>
        const src = msg.src as string | undefined

        if (payload.kind === 'init' && src) {
          // Phone announced itself via relay
          relayPeerIdRef.current = src
          setConnected(true)
          setError(null)
          // Acknowledge so the phone knows relay works
          peer.socket.send({ type: 'RELAY', dst: src, payload: { kind: 'ack' } })
        } else if (payload.kind === 'voice' && typeof payload.text === 'string') {
          if (payload.text && voiceCallbackRef.current) {
            voiceCallbackRef.current(payload.text)
          }
        }
      })
    })

    peer.on('connection', (conn) => {
      connRef.current = conn
      conn.on('open', () => {
        setConnected(true)
        setError(null)
      })

      conn.on('data', (data) => {
        // Phone sends: { type: 'voice', text: '...' }
        if (
          typeof data === 'object' &&
          data !== null &&
          'type' in data &&
          (data as Record<string, unknown>).type === 'voice'
        ) {
          const text = String((data as Record<string, unknown>).text ?? '')
          if (text && voiceCallbackRef.current) {
            voiceCallbackRef.current(text)
          }
        }
      })

      conn.on('close', () => {
        if (!relayPeerIdRef.current) setConnected(false)
        connRef.current = null
      })

      conn.on('error', (err) => {
        setError(`Connection error: ${err.message}`)
      })
    })

    peer.on('error', (err) => {
      setError(`Peer error: ${err.message}`)
    })

    peer.on('disconnected', () => {
      // Try to reconnect to the broker (not the data channel)
      if (!peer.destroyed) {
        peer.reconnect()
      }
    })

    return () => {
      peer.destroy()
    }
  }, [])

  const sendState = useCallback((state: CookingSessionState) => {
    // Prefer WebRTC data channel when available
    const conn = connRef.current
    if (conn?.open) {
      conn.send({ type: 'state', payload: state })
      return
    }
    // Fall back to WebSocket relay through the signaling server
    const dst = relayPeerIdRef.current
    const peer = peerRef.current
    if (dst && peer && !peer.destroyed) {
      peer.socket.send({
        type: 'RELAY',
        dst,
        payload: { kind: 'state', data: state },
      })
    }
  }, [])

  const onVoiceText = useCallback((cb: (text: string) => void) => {
    voiceCallbackRef.current = cb
  }, [])

  const destroy = useCallback(() => {
    connRef.current?.close()
    peerRef.current?.destroy()
    relayPeerIdRef.current = null
    setPeerId(null)
    setConnected(false)
  }, [])

  return { peerId, connected, sendState, onVoiceText, destroy, error }
}
