"""
Data models for the taste engine.

The whole design hinges on ONE intermediate shape that both Spotify input
paths (top-tracks via OAuth, and pasted playlist URL) converge to. The engine
never knows or cares which path produced its input — it only ever sees a list
of ArtistEntry objects. That convergence is what keeps the two input adapters
thin and the analysis logic single-sourced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ArtistEntry:
    """
    One artist as it appears in a user's listening data.

    This is the atomic unit the engine consumes. Path A (/me/top/artists)
    returns these almost directly. Path B (playlist -> tracks -> /artists)
    builds them after hydrating genres from artist IDs. Either way, by the
    time data reaches the engine it looks like this.
    """
    name: str
    genres: list[str] = field(default_factory=list)
    popularity: int = 0            # Spotify's 0-100 popularity score
    release_year: Optional[int] = None  # representative year (track/album)
    weight: float = 1.0            # how much this artist counts (e.g. play
                                   # rank, or track count in a playlist)

    def __post_init__(self) -> None:
        # Normalize genres to lowercase, stripped, de-duped while preserving order.
        seen: set[str] = set()
        cleaned: list[str] = []
        for g in self.genres:
            g_norm = g.strip().lower()
            if g_norm and g_norm not in seen:
                seen.add(g_norm)
                cleaned.append(g_norm)
        self.genres = cleaned
        # Clamp popularity into the valid Spotify range defensively.
        self.popularity = max(0, min(100, int(self.popularity)))
        if self.weight <= 0:
            self.weight = 1.0


@dataclass
class TasteProfile:
    """
    The structured output of the engine. Every field here is something we
    COMPUTED and can explain in one sentence — this is the object the LLM
    narrator receives. The narrator voices these numbers; it never invents
    the analysis.
    """
    # The taste vector: genre -> weighted frequency (sums to 1.0). This is the
    # high-dimensional fingerprint used for cosine similarity / clustering.
    genre_vector: dict[str, float]

    # The four engineered scalar axes, each in [0, 1].
    mainstream_score: float   # 0 = deeply obscure, 1 = pure chart pop
    nostalgia_score: float    # 0 = all brand-new, 1 = all old/classic
    eclectic_score: float     # 0 = laser-focused, 1 = maximally spread (entropy)
    explorer_score: float     # 0 = same few artists, 1 = wide artist spread

    # Supporting numbers the narrator and UI can use.
    top_genres: list[tuple[str, float]]   # genre vector sorted, top N
    distinct_genres: int
    distinct_artists: int
    median_release_year: Optional[int]
    raw_entropy_bits: float               # Shannon entropy before normalizing

    def as_dict(self) -> dict:
        """Flatten to plain JSON-serializable dict for the FastAPI response."""
        return {
            "genre_vector": self.genre_vector,
            "axes": {
                "mainstream": round(self.mainstream_score, 4),
                "nostalgia": round(self.nostalgia_score, 4),
                "eclectic": round(self.eclectic_score, 4),
                "explorer": round(self.explorer_score, 4),
            },
            "top_genres": [
                {"genre": g, "weight": round(w, 4)} for g, w in self.top_genres
            ],
            "distinct_genres": self.distinct_genres,
            "distinct_artists": self.distinct_artists,
            "median_release_year": self.median_release_year,
            "raw_entropy_bits": round(self.raw_entropy_bits, 4),
        }
