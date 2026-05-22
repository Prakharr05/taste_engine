"""
Input adapters: raw Spotify JSON -> the converged ArtistEntry list.

This is where the "both input paths merge" design becomes concrete. Each
adapter is deliberately thin — it does shape conversion only, no analysis.
Whatever produced the data, the engine downstream sees identical input.

NOTE: these take ALREADY-FETCHED Spotify JSON. The actual HTTP calls + OAuth
live in the Next.js frontend (Phase 3); FastAPI receives the fetched payloads.
Keeping the network out of here makes the adapters trivially unit-testable.
"""

from __future__ import annotations

from typing import Any

from engine import ArtistEntry


def from_top_artists(payload: dict[str, Any]) -> list[ArtistEntry]:
    """
    Path A: Spotify /me/top/artists response.

    Artists arrive with genres already attached, so this is the cleanest path.
    We weight by rank position — higher-ranked artists count for more, which
    is a better signal of taste than treating every artist equally.
    """
    items = payload.get("items", [])
    n = len(items)
    entries: list[ArtistEntry] = []
    for rank, artist in enumerate(items):
        # Linear rank weighting: #1 gets weight n, last gets weight 1.
        weight = n - rank
        entries.append(ArtistEntry(
            name=artist.get("name", "Unknown"),
            genres=artist.get("genres", []),
            popularity=artist.get("popularity", 0),
            release_year=None,  # top-artists has no per-artist year; that's fine
            weight=float(weight),
        ))
    return entries


def from_playlist(tracks_payload: dict[str, Any],
                  artists_payload: dict[str, Any]) -> list[ArtistEntry]:
    """
    Path B: a public playlist.

    Playlist tracks give us artist IDs and release years but NOT genres, so the
    frontend also batch-fetches /artists?ids=... and passes both payloads here.
    We:
      1. count how many tracks each artist appears on (-> weight)
      2. capture the earliest/representative release year per artist
      3. join genres + popularity from the artists payload
    Deduping by artist is what makes the later genre-fetch cheap and cacheable.
    """
    # 1. tally tracks per artist + collect release years
    track_count: dict[str, int] = {}
    years: dict[str, list[int]] = {}

    for item in tracks_payload.get("items", []):
        track = item.get("track") or {}
        year = _parse_year(track.get("album", {}).get("release_date"))
        for artist in track.get("artists", []):
            aid = artist.get("id")
            if not aid:
                continue
            track_count[aid] = track_count.get(aid, 0) + 1
            if year is not None:
                years.setdefault(aid, []).append(year)

    # 2. index the artists payload by id for genre + popularity lookup
    artist_info = {a["id"]: a for a in artists_payload.get("artists", []) if a.get("id")}

    # 3. build entries
    entries: list[ArtistEntry] = []
    for aid, count in track_count.items():
        info = artist_info.get(aid, {})
        rep_year = min(years[aid]) if years.get(aid) else None
        entries.append(ArtistEntry(
            name=info.get("name", "Unknown"),
            genres=info.get("genres", []),
            popularity=info.get("popularity", 0),
            release_year=rep_year,
            weight=float(count),
        ))
    return entries


def collect_artist_ids(tracks_payload: dict[str, Any]) -> list[str]:
    """
    Helper for the frontend: pull the DEDUPED artist IDs from a playlist so it
    knows which IDs to batch into /artists?ids=... (max 50 per call). Returned
    here so the dedup logic lives next to the adapter that consumes it.
    """
    seen: list[str] = []
    seen_set: set[str] = set()
    for item in tracks_payload.get("items", []):
        track = item.get("track") or {}
        for artist in track.get("artists", []):
            aid = artist.get("id")
            if aid and aid not in seen_set:
                seen_set.add(aid)
                seen.append(aid)
    return seen


def _parse_year(release_date: str | None) -> int | None:
    """Spotify release_date is 'YYYY', 'YYYY-MM', or 'YYYY-MM-DD'."""
    if not release_date:
        return None
    try:
        return int(release_date[:4])
    except (ValueError, TypeError):
        return None
