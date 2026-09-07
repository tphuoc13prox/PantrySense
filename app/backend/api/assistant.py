from __future__ import annotations

import re
from typing import Any
from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from app.backend.recipes import substitution
from app.backend.recipes.schemas import CookingStepInfo, ExtractedTimer

router = APIRouter(prefix="/api/assistant", tags=["assistant"])


class ParseStepsRequest(BaseModel):
    instructions: list[str] = Field(default_factory=list)


def extract_timer_from_instruction(text: str) -> tuple[int | None, str | None]:
    """Extracts duration in minutes from recipe instruction text."""
    # Look for hour patterns (e.g. "1 hour", "1.5 hours", "2 hrs")
    hour_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:-|–|to\s*\d+\s*)?\s*(?:hours?|hrs?)\b", text, re.I)
    if hour_match:
        try:
            hrs = float(hour_match.group(1))
            mins = int(hrs * 60)
            return mins, f"{hour_match.group(0).strip()} timer"
        except Exception:
            pass

    # Look for minute patterns (e.g. "10 minutes", "5-7 mins", "15 min")
    min_match = re.search(r"(\d+)(?:\s*(?:-|–|to)\s*(\d+))?\s*(?:minutes?|mins?|min)\b", text, re.I)
    if min_match:
        try:
            # If a range like "10 to 15 mins" is found, pick the higher value
            mins = int(min_match.group(2) or min_match.group(1))
            return mins, f"{min_match.group(0).strip()} timer"
        except Exception:
            pass

    return None, None


@router.post("/parse-steps", response_model=list[CookingStepInfo])
def parse_cooking_steps(payload: ParseStepsRequest) -> list[CookingStepInfo]:
    """Parses instructions into sequential steps with auto-detected timer durations."""
    steps: list[CookingStepInfo] = []
    for idx, inst in enumerate(payload.instructions, start=1):
        clean_text = inst.strip()
        mins, desc = extract_timer_from_instruction(clean_text)
        steps.append(
            CookingStepInfo(
                step_number=idx,
                instruction=clean_text,
                timer_minutes=mins,
                timer_description=desc,
            )
        )
    return steps


@router.get("/substitutions")
def get_ingredient_substitutions(
    ingredient: str = Query(..., description="Name of the ingredient to substitute")
) -> dict[str, Any]:
    """Looks up available culinary substitutes for any ingredient."""
    subs = substitution.get_substitutions_for_ingredient(ingredient)
    return {
        "ingredient": ingredient,
        "substitutes": subs,
    }
