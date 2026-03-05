"""
Minimal PeerJS-compatible signaling server.

Replaces the public PeerJS cloud broker (0.peerjs.com) with a self-hosted
WebSocket relay running inside the existing FastAPI process.  Both the PC
host and the phone remote connect here for WebRTC signaling; once the peer
connection is established all data flows directly P2P.

Protocol (mirrors peerjs-server):
  • Client opens  WS /peerjs?key=peerjs&id=<id>&token=<token>
  • Server sends  {"type":"OPEN"}
  • Client sends  {"type":"OFFER|ANSWER|CANDIDATE","dst":"<target>","payload":{…}}
  • Server relays {"type":"OFFER|ANSWER|CANDIDATE","src":"<sender>","dst":"<target>","payload":{…}}
  • Client sends  {"type":"HEARTBEAT"}  (keep-alive, not relayed)
"""

from __future__ import annotations

import json
import logging
import uuid

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["signaling"])

# ── Connected peers ────────────────────────────────────────────────
_peers: dict[str, WebSocket] = {}


@router.get("/peerjs/id", response_class=PlainTextResponse)
async def generate_peer_id():
    """Return a random peer ID for clients that don't supply their own."""
    return uuid.uuid4().hex[:16]


@router.get("/peerjs/peers")
async def list_peers():
    """Return the list of currently connected peer IDs (debug helper)."""
    return list(_peers.keys())


@router.websocket("/peerjs")
async def peerjs_signaling(
    websocket: WebSocket,
    key: str = Query(default="peerjs"),
    id: str = Query(default=""),
    token: str = Query(default=""),
):
    """PeerJS-compatible WebSocket signaling endpoint."""
    logger.info("Signaling: WS request for id='%s' key='%s'", id, key)

    if not id:
        await websocket.close(code=4000, reason="Missing peer id")
        return

    # Reject duplicate IDs
    if id in _peers:
        await websocket.accept()
        await websocket.send_text(
            json.dumps(
                {"type": "ID-TAKEN", "payload": {"msg": f'ID "{id}" is already taken'}}
            )
        )
        await websocket.close()
        return

    await websocket.accept()
    _peers[id] = websocket
    logger.info("Signaling: peer '%s' connected (%d total)", id, len(_peers))

    # Confirm registration — PeerJS client waits for this before emitting 'open'
    try:
        await websocket.send_text(json.dumps({"type": "OPEN"}))
    except Exception:
        _peers.pop(id, None)
        return

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            msg_type = msg.get("type")

            if msg_type == "HEARTBEAT":
                continue

            if msg_type == "LEAVE":
                break

            dst = msg.get("dst")
            if not dst:
                continue

            target_ws = _peers.get(dst)
            if target_ws is None:
                # Destination not connected — inform the sender
                try:
                    await websocket.send_text(
                        json.dumps({"type": "LEAVE", "src": dst})
                    )
                except Exception:
                    break
                continue

            # Forward with sender identity attached
            forward = {**msg, "src": id}
            try:
                await target_ws.send_text(json.dumps(forward))
            except Exception:
                # Target dropped
                _peers.pop(dst, None)

    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("Signaling: unexpected error for peer '%s'", id)
    finally:
        _peers.pop(id, None)
        logger.info(
            "Signaling: peer '%s' disconnected (%d remaining)", id, len(_peers)
        )
