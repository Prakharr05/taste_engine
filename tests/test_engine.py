"""
Tests for the taste engine. These check the *properties* of each measure
(monotonicity, bounds, known edge cases) rather than hardcoded magic numbers,
so they document what each axis is supposed to mean and stay robust to tuning.
"""

import math

import pytest

from engine import ArtistEntry, analyze, compatibility
from engine.analyzer import (
    build_genre_vector,
    eclectic_score,
    explorer_score,
    mainstream_score,
    nostalgia_score,
    shannon_entropy_bits,
)


def make(name, genres, pop=50, year=2020, weight=1.0):
    return ArtistEntry(name=name, genres=genres, popularity=pop,
                       release_year=year, weight=weight)


# ---- genre vector --------------------------------------------------------

def test_genre_vector_sums_to_one():
    artists = [make("A", ["pop"]), make("B", ["rock", "indie"])]
    vec = build_genre_vector(artists)
    assert pytest.approx(sum(vec.values()), abs=1e-9) == 1.0


def test_genre_weight_splits_across_an_artists_genres():
    # One artist, two genres, weight 1 -> each genre gets 0.5.
    vec = build_genre_vector([make("A", ["pop", "rock"], weight=1.0)])
    assert vec["pop"] == pytest.approx(0.5)
    assert vec["rock"] == pytest.approx(0.5)


def test_artist_with_no_genres_contributes_nothing_to_vector():
    vec = build_genre_vector([make("A", ["pop"]), make("B", [])])
    assert set(vec) == {"pop"}


# ---- entropy / eclectic --------------------------------------------------

def test_entropy_zero_for_single_genre():
    assert shannon_entropy_bits({"pop": 1.0}) == 0.0


def test_entropy_matches_known_value_for_uniform_two():
    # Two equally-likely outcomes => exactly 1 bit.
    assert shannon_entropy_bits({"a": 0.5, "b": 0.5}) == pytest.approx(1.0)


def test_eclectic_focused_user_lower_than_eclectic_user():
    focused = [make("A", ["pop"]), make("B", ["pop"]), make("C", ["pop"])]
    eclectic = [make("A", ["pop"]), make("B", ["jazz"]), make("C", ["metal"])]
    assert eclectic_score(build_genre_vector(focused)) < \
           eclectic_score(build_genre_vector(eclectic))


def test_eclectic_is_bounded():
    eclectic = [make("A", ["pop"]), make("B", ["jazz"]), make("C", ["metal"])]
    score = eclectic_score(build_genre_vector(eclectic))
    assert 0.0 <= score <= 1.0


# ---- mainstream ----------------------------------------------------------

def test_mainstream_high_for_popular_artists():
    artists = [make("A", ["pop"], pop=95), make("B", ["pop"], pop=90)]
    assert mainstream_score(artists) > 0.85


def test_mainstream_low_for_obscure_artists():
    artists = [make("A", ["pop"], pop=10), make("B", ["pop"], pop=5)]
    assert mainstream_score(artists) < 0.15


def test_mainstream_respects_weight():
    # Heavy weight on the obscure artist should drag the score down.
    artists = [make("A", ["pop"], pop=100, weight=1),
               make("B", ["pop"], pop=0, weight=9)]
    assert mainstream_score(artists) < 0.2


# ---- nostalgia -----------------------------------------------------------

def test_nostalgia_high_for_old_music():
    artists = [make("A", ["rock"], year=1975), make("B", ["rock"], year=1980)]
    assert nostalgia_score(artists, now_year=2025) > 0.8


def test_nostalgia_low_for_new_music():
    artists = [make("A", ["pop"], year=2024), make("B", ["pop"], year=2025)]
    assert nostalgia_score(artists, now_year=2025) < 0.1


def test_nostalgia_zero_when_no_years():
    a = ArtistEntry(name="A", genres=["pop"], popularity=50, release_year=None)
    assert nostalgia_score([a], now_year=2025) == 0.0


# ---- explorer ------------------------------------------------------------

def test_explorer_low_when_one_artist_dominates():
    artists = [make("A", ["pop"], weight=100)] + \
              [make(f"X{i}", ["pop"], weight=1) for i in range(5)]
    assert explorer_score(artists) < 0.3


def test_explorer_high_when_evenly_spread():
    artists = [make(f"X{i}", ["pop"], weight=1) for i in range(10)]
    assert explorer_score(artists) > 0.9


# ---- compatibility -------------------------------------------------------

def test_identical_profiles_are_maximally_compatible():
    artists = [make("A", ["pop"], pop=80, year=2020),
               make("B", ["rock"], pop=60, year=2018)]
    p = analyze(artists)
    comp = compatibility(p, p)
    assert comp["overall"] == pytest.approx(1.0, abs=1e-6)
    assert comp["genre_similarity"] == pytest.approx(1.0, abs=1e-6)


def test_disjoint_genres_have_zero_genre_similarity():
    p1 = analyze([make("A", ["polka"], pop=50, year=2020)])
    p2 = analyze([make("B", ["death metal"], pop=50, year=2020)])
    comp = compatibility(p1, p2)
    assert comp["genre_similarity"] == pytest.approx(0.0)
    # But axis closeness can still be high (same pop/year), so overall > 0.
    assert comp["overall"] > 0.0


# ---- end to end ----------------------------------------------------------

def test_analyze_produces_full_profile():
    artists = [make("A", ["indie folk", "bedroom pop"], pop=60, year=2019, weight=3),
               make("B", ["shoegaze"], pop=45, year=1993, weight=2)]
    p = analyze(artists)
    d = p.as_dict()
    assert d["distinct_artists"] == 2
    assert set(d["axes"]) == {"mainstream", "nostalgia", "eclectic", "explorer"}
    for v in d["axes"].values():
        assert 0.0 <= v <= 1.0


def test_analyze_rejects_empty_input():
    with pytest.raises(ValueError):
        analyze([])
