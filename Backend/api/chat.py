"""
Product / recipe chat endpoint.

POST /chat
    Body (JSON):
      - message:  str           (user's question)
      - context:  dict | None   (product/recipe data for grounding)
      - history:  list[dict]    (previous messages in the conversation)

Returns a plain-text LLM reply grounded in the supplied food context.
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from Backend.dependencies.router import get_router

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Chat"])


# ── Request / Response schemas ──────────────────────────────────────

class ChatMessage(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User's question")
    context: Optional[Dict[str, Any]] = Field(
        None,
        description="Product or recipe data to ground the conversation",
    )
    history: List[ChatMessage] = Field(
        default_factory=list,
        description="Previous messages for multi-turn context",
    )


class ChatResponse(BaseModel):
    reply: str


# ── Endpoint ────────────────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest, nutri_router=Depends(get_router)):
    """
    Conversational chat about a specific food item.

    The LLM is grounded with the ``context`` dict (nutrition, ingredients,
    brand, category, etc.) so it can answer accurately without hallucinating.
    """

    # Build a grounding block from the context dict
    ctx_block = ""
    if body.context:
        lines = []
        for k, v in body.context.items():
            if v is not None and v != "":
                lines.append(f"  {k}: {v}")
        if lines:
            ctx_block = "FOOD ITEM CONTEXT:\n" + "\n".join(lines)

    # Build conversation history block
    history_block = ""
    if body.history:
        turns = []
        for msg in body.history[-6:]:  # keep last 6 messages to stay within context window
            prefix = "User" if msg.role == "user" else "Assistant"
            turns.append(f"{prefix}: {msg.content}")
        if turns:
            history_block = "CONVERSATION HISTORY:\n" + "\n".join(turns)

    prompt = f"""You are NutriSense AI, a knowledgeable nutrition assistant specialising in Indian cuisine and food products.

{ctx_block}

{history_block}

USER QUESTION: {body.message}

RULES:
- Answer the user's question accurately using the provided food context.
- If specific nutrition data is available in the context, reference those exact values.
- Keep responses concise but informative (2-4 paragraphs max).
- If the user asks something outside the provided context, state that clearly.
- Do NOT invent nutrition numbers that are not in the context.
- Be conversational and helpful.
"""

    try:
        llm_engine = nutri_router.llm_engine
        reply = await llm_engine.llm.generate_async(prompt)
        return ChatResponse(reply=reply)
    except Exception as exc:
        logger.exception("Chat generation failed: %s", exc)
        raise HTTPException(status_code=500, detail="Chat service unavailable. Please try again.")
