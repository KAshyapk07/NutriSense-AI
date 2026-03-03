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

export function usePeerConnection(): PeerHookReturn {
  const [peerId, setPeerId] = useState<string | null>(null)
  const [connected, setConnected] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const peerRef = useRef<Peer | null>(null)
  const connRef = useRef<DataConnection | null>(null)
  const voiceCallbackRef = useRef<((text: string) => void) | null>(null)

  // Initialize peer on mount
  useEffect(() => {
    const id = `nutri-chef-${randomSuffix()}`
    const peer = new Peer(id, {
      // Use the free PeerJS cloud broker (outbound only — firewall-safe)
      // No config needed; PeerJS defaults to 0.peerjs.com
      debug: 0, // silent in production
    })

    peerRef.current = peer

    peer.on('open', (assignedId) => {
      setPeerId(assignedId)
      setError(null)
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
        setConnected(false)
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
    const conn = connRef.current
    if (conn?.open) {
      conn.send({ type: 'state', payload: state })
    }
  }, [])

  const onVoiceText = useCallback((cb: (text: string) => void) => {
    voiceCallbackRef.current = cb
  }, [])

  const destroy = useCallback(() => {
    connRef.current?.close()
    peerRef.current?.destroy()
    setPeerId(null)
    setConnected(false)
  }, [])

  return { peerId, connected, sendState, onVoiceText, destroy, error }
}
