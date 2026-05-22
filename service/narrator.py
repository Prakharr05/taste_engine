"""
The narrator: turns a computed TasteProfile into the shareable roast.

THE ONE RULE THIS FILE ENFORCES: the LLM sees only the *numbers* the engine
computed — axes, top genres, entropy, counts. It never sees the raw track or
artist list. It therefore cannot invent an analysis; it can only voice ours.
That is the entire integrity argument of the project, enforced here by what we
put in the prompt (and what we leave out).

Provider-agnostic: set LLM_PROVIDER + the matching API key and it calls the
real model. With no key it falls back to a deterministic template narrator so
the service runs end-to-end offline (useful for tests, CI, and demos).
"""

from __future__ import annotations

import json
import os
from typing import Any

from engine import TasteProfile

# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a witty, perceptive music-taste analyst. You write \
short, punchy personality readings based ONLY on the numeric profile you are \
given. You never invent specific songs, artists, or facts that are not present \
in the data. You are playful and a little roasty, but never mean about \
protected characteristics. Tease the listener's habits and quirks, never their \
worth — taste is close to identity, so roast the behavior ('you'll call music \
textural'), not the person ('your taste is embarrassing'). The reading should \
be something the listener would happily screenshot and share. Output ONLY valid \
JSON matching the requested schema, with no markdown fences or preamble."""

# What the model is allowed to know. Note what is ABSENT: no track names, no
# artist names beyond the genre labels, nothing it could spin a fake story from.
SCHEMA_INSTRUCTION = """Return a JSON object with exactly these keys:
{
  "personality_type": "a 2-4 word archetype name, title case",
  "summary": "2-3 sentence personality read grounded in the numbers",
  "green_flags": ["3 short positive observations"],
  "red_flags": ["2-3 short playful warnings"],
  "dating_verdict": "1-2 sentences on what dating this listener is like"
}"""


def _describe_axis(name: str, value: float, low: str, high: str) -> str:
    """Turn a 0-1 axis into a plain-language band the model can reason over."""
    if value < 0.2:
        band = f"very {low}"
    elif value < 0.4:
        band = f"somewhat {low}"
    elif value < 0.6:
        band = "balanced"
    elif value < 0.8:
        band = f"somewhat {high}"
    else:
        band = f"very {high}"
    return f"{name}: {value:.2f} ({band})"


def build_user_prompt(profile: TasteProfile) -> str:
    """
    Render the computed profile into the text the model receives. This is the
    ONLY information about the listener that crosses into the LLM.
    """
    axes = [
        _describe_axis("mainstream", profile.mainstream_score, "obscure", "mainstream"),
        _describe_axis("nostalgia", profile.nostalgia_score, "current", "nostalgic"),
        _describe_axis("eclectic", profile.eclectic_score, "focused", "eclectic"),
        _describe_axis("explorer", profile.explorer_score, "loyal to favorites", "always exploring"),
    ]
    genres = ", ".join(f"{g} ({w:.0%})" for g, w in profile.top_genres)

    return (
        "Here is the computed taste profile. Base your reading ONLY on this:\n\n"
        f"AXES (each 0.0-1.0):\n  " + "\n  ".join(axes) + "\n\n"
        f"TOP GENRES (by weight): {genres}\n"
        f"distinct genres: {profile.distinct_genres}\n"
        f"distinct artists: {profile.distinct_artists}\n"
        f"median release year: {profile.median_release_year}\n\n"
        + SCHEMA_INSTRUCTION
    )


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------

def _call_anthropic(system: str, user: str) -> str:
    import anthropic  # imported lazily so the package is optional

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    model = os.getenv("LLM_MODEL", "claude-sonnet-4-6")
    resp = client.messages.create(
        model=model,
        max_tokens=700,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(block.text for block in resp.content if block.type == "text")


def _call_openai(system: str, user: str) -> str:
    from openai import OpenAI  # imported lazily

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    resp = client.chat.completions.create(
        model=model,
        max_tokens=700,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content or ""


def _fallback_narrator(profile: TasteProfile) -> dict[str, Any]:
    """
    Deterministic template narrator used when no LLM key is configured. Keeps
    the whole service runnable offline. It reads the same numbers the LLM would,
    so the contract (numbers in -> roast out) is identical; only the prose
    quality differs.
    """
    top_genre = profile.top_genres[0][0] if profile.top_genres else "music"

    if profile.mainstream_score > 0.7:
        type_word = "Chart Devotee"
        main_line = "You like what everyone likes, and you like it loudly."
    elif profile.mainstream_score < 0.3:
        type_word = "Deep Cut Archivist"
        main_line = "If a song hits 50 popularity you've probably already left."
    else:
        type_word = "Balanced Listener"
        main_line = "You straddle the charts and the underground without apology."

    eclectic_note = ("genre-hopping" if profile.eclectic_score > 0.6
                     else "loyal to a tight lane")
    era_note = ("living in the past" if profile.nostalgia_score > 0.6
                else "firmly in the now")

    return {
        "personality_type": type_word,
        "summary": (
            f"A {eclectic_note} listener anchored in {top_genre}, "
            f"{era_note}. {main_line}"
        ),
        "green_flags": [
            f"Strong point of view: {top_genre} is clearly home base.",
            f"{'Adventurous ears' if profile.explorer_score > 0.6 else 'Knows what they love'}.",
            f"{'Refreshingly current' if profile.nostalgia_score < 0.4 else 'Respects the classics'}.",
        ],
        "red_flags": [
            f"{'May overwhelm you with playlists' if profile.explorer_score > 0.7 else 'Might replay the same five songs forever'}.",
            f"{'Could be a music snob' if profile.mainstream_score < 0.3 else 'Will defend a guilty pleasure to the death'}.",
        ],
        "dating_verdict": (
            f"Dating a {type_word} means {'a constant stream of song recommendations' if profile.explorer_score > 0.6 else 'a comfortable, predictable soundtrack'}. "
            f"{'Bring an open mind.' if profile.eclectic_score > 0.6 else 'Bring patience for repeats.'}"
        ),
        "_fallback": True,
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def narrate(profile: TasteProfile) -> dict[str, Any]:
    """
    Produce the roast. Routes to the configured LLM provider, falling back to
    the deterministic narrator if no key is set or the call/parse fails.
    """
    provider = os.getenv("LLM_PROVIDER", "").lower()
    debug = os.getenv("NARRATOR_DEBUG", "").lower() in ("1", "true", "yes")
    user_prompt = build_user_prompt(profile)

    caller = None
    if provider == "anthropic" and os.getenv("ANTHROPIC_API_KEY"):
        caller = _call_anthropic
    elif provider == "openai" and os.getenv("OPENAI_API_KEY"):
        caller = _call_openai

    if caller is None:
        # No provider configured. In debug mode, say WHY so a misconfigured
        # deploy doesn't silently serve template roasts forever.
        if debug:
            print(f"[narrator] falling back: no caller "
                  f"(provider={provider!r}, "
                  f"anthropic_key={bool(os.getenv('ANTHROPIC_API_KEY'))}, "
                  f"openai_key={bool(os.getenv('OPENAI_API_KEY'))})")
        result = _fallback_narrator(profile)
        if debug:
            result["_fallback_reason"] = "no_provider_configured"
        return result

    try:
        raw = caller(SYSTEM_PROMPT, user_prompt)
        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        result = json.loads(cleaned)
        result["_fallback"] = False
        return result
    except Exception as exc:
        # Degrade gracefully so a model hiccup never 500s the endpoint — but in
        # debug mode surface the real error instead of hiding it. The silent
        # version of this cost real debugging time; never again on a deploy.
        if debug:
            import traceback
            traceback.print_exc()
        result = _fallback_narrator(profile)
        if debug:
            result["_fallback_reason"] = f"{type(exc).__name__}: {exc}"
        return result