from __future__ import annotations

import re
from typing import Any
from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from app.backend.recipes import substitution
from app.backend.recipes.instructions import clean_and_segment_instructions
from app.backend.recipes.schemas import CookingStepInfo, ExtractedTimer

router = APIRouter(prefix="/api/assistant", tags=["assistant"])


class ParseStepsRequest(BaseModel):
    instructions: list[str] | str = Field(default_factory=list)


def extract_timer_from_instruction(text: str) -> tuple[int | None, str | None]:
    """Extracts duration in minutes from recipe instruction text."""
    # 1. Combined hour and minute pattern (e.g. "1 hour 30 minutes", "1 hr 15 mins", "1 hour and 20 min")
    compound_match = re.search(
        r"(\d+)\s*(?:hours?|hrs?)\s*(?:and)?\s*(\d+)\s*(?:minutes?|mins?|min)\b",
        text,
        re.I,
    )
    if compound_match:
        try:
            hrs = int(compound_match.group(1))
            mins = int(compound_match.group(2))
            total = hrs * 60 + mins
            return total, f"{hrs} hr {mins} min timer"
        except Exception:
            pass

    # 2. Fractions of an hour (e.g. "1/2 hour", "1 1/2 hours", "3/4 hour")
    frac_hour = re.search(r"(\d+\s+)?(\d+)/(\d+)\s*(?:hours?|hrs?)\b", text, re.I)
    if frac_hour:
        try:
            whole = int(frac_hour.group(1).strip()) if frac_hour.group(1) else 0
            num = int(frac_hour.group(2))
            den = int(frac_hour.group(3))
            total_mins = int((whole + num / den) * 60)
            return total_mins, f"{frac_hour.group(0).strip()} timer"
        except Exception:
            pass

    # 3. Hour patterns (e.g. "1 hour", "1.5 hours", "2 hrs", "2-3 hours")
    hour_match = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:-|–|to\s*(\d+(?:\.\d+)?)\s*)?\s*(?:hours?|hrs?)\b",
        text,
        re.I,
    )
    if hour_match:
        try:
            val = float(hour_match.group(2) or hour_match.group(1))
            mins = int(val * 60)
            return mins, f"{hour_match.group(0).strip()} timer"
        except Exception:
            pass

    # 4. Minute patterns (e.g. "10 minutes", "5-7 mins", "15 min", "20 to 25 minutes")
    min_match = re.search(
        r"(\d+)(?:\s*(?:-|–|to)\s*(\d+))?\s*(?:minutes?|mins?|min)\b",
        text,
        re.I,
    )
    if min_match:
        try:
            mins = int(min_match.group(2) or min_match.group(1))
            return mins, f"{min_match.group(0).strip()} timer"
        except Exception:
            pass

    return None, None


@router.post("/parse-steps", response_model=list[CookingStepInfo])
def parse_cooking_steps(payload: ParseStepsRequest) -> list[CookingStepInfo]:
    """Parses instructions into sequential steps with auto-detected timer durations."""
    clean_instructions = clean_and_segment_instructions(payload.instructions)
    steps: list[CookingStepInfo] = []
    for idx, inst in enumerate(clean_instructions, start=1):
        mins, desc = extract_timer_from_instruction(inst)
        steps.append(
            CookingStepInfo(
                step_number=idx,
                instruction=inst,
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
