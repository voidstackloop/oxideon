"""Tests for data curation (oxideon.curate)."""
import numpy as np
import pytest

import oxideon as ox
from oxideon import (
    MinHashLSH,
    NGramDecontaminator,
    c4_flags,
    doremi_weights,
    exact_jaccard,
    gopher_flags,
    passes_c4,
    passes_gopher,
    regmix_fit,
    regmix_search,
    temperature_mix,
)


def test_top_level_exports():
    assert ox.MinHashLSH is MinHashLSH
    assert ox.NGramDecontaminator is NGramDecontaminator
    assert ox.curate.doremi_weights is doremi_weights


def test_exact_jaccard():
    assert exact_jaccard("abcde", "abcde") == 1.0
    assert exact_jaccard("abcde", "abcdf", ngram=2) == pytest.approx(0.6)
    assert exact_jaccard("aaaa", "zzzz") == 0.0
    assert exact_jaccard("", "") == 1.0
    with pytest.raises(ValueError):
        exact_jaccard("a", "b", ngram=0)


def test_minhash_finds_duplicates():
    lsh = MinHashLSH(num_perm=128, bands=32, seed=0)
    base = " ".join(f"word{i}" for i in range(50))
    near = " ".join(f"word{i}" if i != 25 else "changed" for i in range(50))
    other = "completely different document about nothing shared whatsoever here"
    lsh.add("a", base)
    lsh.add("b", base)
    lsh.add("c", near)
    lsh.add("d", other)
    assert set(lsh.query(base)) == {"a", "b", "c"}
    clusters = lsh.duplicates()
    assert ["a", "b", "c"] in clusters
    assert all("d" not in cluster for cluster in clusters)
    assert len(lsh.signature(base)) == 128
    lsh.add("e", "")
    lsh.add("f", "")
    assert any(set(c) == {"e", "f"} for c in lsh.duplicates())


def test_minhash_validation():
    with pytest.raises(ValueError):
        MinHashLSH(num_perm=100, bands=16)
    with pytest.raises(ValueError):
        MinHashLSH(seed=-1)
    lsh = MinHashLSH()
    lsh.add("a", "hello world")
    with pytest.raises(ValueError):
        lsh.add("a", "hello again")


def test_decontaminator():
    eval_text = "the quick brown fox jumps over the lazy dog and then runs away fast"
    dec = NGramDecontaminator(n=13).fit([eval_text])
    assert dec.is_contaminated(eval_text)
    assert dec.is_contaminated("prefix words " + eval_text + " suffix words")
    assert not dec.is_contaminated("something entirely different about cooking pasta")
    assert dec.contamination_rate([eval_text, "fresh unrelated sentence here"]) == 0.5
    with pytest.raises(ValueError):
        NGramDecontaminator(n=0)
    with pytest.raises(ValueError):
        NGramDecontaminator().fit([])
    with pytest.raises(ValueError):
        dec.contamination_rate([])


def _good_text():
    words = [f"token{i:03d}" for i in range(72)]
    lines = [" ".join(words[i * 12 : (i + 1) * 12]) + "." for i in range(6)]
    return "\n".join(lines)


def test_gopher_flags():
    assert passes_gopher(_good_text())
    assert all(gopher_flags(_good_text()).values())
    short = "too few words here"
    assert gopher_flags(short)["word_count_ok"] is False
    tiny = " ".join(["a"] * 60)
    assert gopher_flags(tiny)["mean_word_length_ok"] is False
    assert gopher_flags(_good_text() + " # " * 30)["hash_ratio_ok"] is False
    bullets = "\n".join(["- item"] * 10 + ["plain"])
    assert gopher_flags(bullets)["bullet_ok"] is False
    ellips = "\n".join(["trailing..."] * 4 + ["fine"] * 6)
    assert gopher_flags(ellips)["ellipsis_line_ok"] is False
    repeat = "\n".join(["same line here"] * 10)
    assert gopher_flags(repeat)["dup_line_ok"] is False


def test_c4_flags():
    assert passes_c4("Hello world, this is fine.")
    assert c4_flags("no terminal punctuation here")["terminal_punct_ok"] is False
    assert c4_flags("too short.")["min_words_ok"] is False
    assert c4_flags("Lorem ipsum dolor sit amet here.")["no_lorem_ok"] is False
    assert c4_flags("Some { code } here now.")["no_code_braces_ok"] is False
    assert c4_flags("This porn content is here now.")["no_badwords_ok"] is False


def test_doremi_weights():
    import math

    assert doremi_weights([1.0, 0.0], [0.5, 0.5], math.log(2)) == pytest.approx([2 / 3, 1 / 3])
    assert doremi_weights([0.5, 0.5], step_size=0.0) == pytest.approx([0.5, 0.5])
    assert sum(doremi_weights([2.0, 1.0, 0.0])) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        doremi_weights([1.0], [0.5, 0.5])
    with pytest.raises(ValueError):
        doremi_weights([1.0, float("nan")])
    with pytest.raises(ValueError):
        doremi_weights([1.0, 0.0], step_size=-1.0)


def test_regmix_fit_and_search():
    mixtures = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.75, 0.25]]
    losses = [0.0, 1.0, 0.5, 0.25]
    coef, intercept = regmix_fit(mixtures, losses, l2=1e-6)
    preds = [sum(c * m for c, m in zip(coef, mix)) + intercept for mix in mixtures]
    assert preds == pytest.approx(losses, abs=1e-3)
    best, pred = regmix_search(coef, intercept, n_candidates=2000, seed=0)
    assert best[0] > 0.8
    assert sum(best) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        regmix_fit([[0.5, 0.5], [0.2, 0.2]], [0.1, 0.2])
    with pytest.raises(ValueError):
        regmix_fit([[1.0, 0.0]], [0.1, 0.2])


def test_temperature_mix():
    assert temperature_mix([100.0, 1.0], 1.0) == pytest.approx([100 / 101, 1 / 101])
    flat = temperature_mix([100.0, 1.0], 100.0)
    assert all(abs(p - 0.5) < 0.05 for p in flat)
    with pytest.raises(ValueError):
        temperature_mix([1.0, 1.0], 0.0)
    with pytest.raises(ValueError):
        temperature_mix([0.0, 0.0], 1.0)
