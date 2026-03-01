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

    prompt = f"""You are NutriSense AI — an expert-level nutritionist and food scientist specialising in Indian cuisine, packaged food products, and evidence-based dietary advice.

You are having a dedicated conversation about a specific food item. All known facts about this item are provided below as structured context. Use this data as your primary source of truth.

{ctx_block}

{history_block}

USER QUESTION: {body.message}

INSTRUCTIONS:
1. ACCURACY FIRST: When the context provides specific nutrition values (calories, protein, carbs, fats, fibre, sodium, etc.), cite those exact numbers in your answer. Never fabricate values.
2. CONTEXTUAL DEPTH: If the user asks about health implications, dietary suitability, or comparisons, reason from the provided data. For example, if protein is 25g, you can confirm it qualifies as a "high protein" option.
3. INDIAN CUISINE EXPERTISE: You understand regional Indian cooking techniques, common ingredient substitutions, traditional preparations, and how they affect nutritional profiles.
4. DIETARY GUIDANCE: When asked about suitability for specific diets (keto, diabetic-friendly, vegan, etc.), evaluate based on the actual macro/micronutrient data available.
5. HONEST BOUNDARIES: If the user's question falls outside the provided context, acknowledge this clearly. Say what you can infer and what would require additional data.
6. STRUCTURE: Use clear paragraphs. For nutrition comparisons or breakdowns, present data in a readable format. Keep responses concise but thorough (2-5 paragraphs).
7. TONE: Professional, warm, and authoritative — like a knowledgeable dietitian speaking to a client.
8. If asked about recipe modifications, suggest concrete ingredient swaps or technique changes, explaining the nutritional impact of each change.
9. Never start with "Based on the context provided" or similar meta-phrases. Speak directly about the food item by name."""

    try:
        llm_engine = nutri_router.engine
        reply = await llm_engine.llm.generate_async(prompt)
        return ChatResponse(reply=reply)
    except Exception as exc:
        logger.exception("Chat generation failed: %s", exc)
        raise HTTPException(status_code=500, detail="Chat service unavailable. Please try again.")
