"""Tests for the offline eval harness (oxideon.evals)."""
import math

import pytest

import oxideon as ox
from oxideon.evals import (
    ChoiceItem,
    CodeItem,
    CodeTask,
    EvalReport,
    MathItem,
    MathTask,
    MultipleChoiceTask,
    NeedleTask,
    accuracy,
    corpus_bleu,
    demo_choices,
    demo_code,
    demo_math,
    exact_match,
    load_jsonl,
    make_haystack,
    math_equal,
    normalize_answer,
    pass_at_k,
    perplexity_from_nll,
    rouge_l,
    run_suite,
    run_untrusted_tests,
    token_f1,
)


class FakeSeq:
    """Generates canned outputs in order, padding with '' when exhausted."""

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def generate(self, prompts):
        self.calls.append(list(prompts))
        out = self.outputs[: len(prompts)]
        self.outputs = self.outputs[len(prompts) :]
        while len(out) < len(prompts):
            out.append("")
        return out


class FakeLoglik:
    def __init__(self, best=" A"):
        self.best = best

    def generate(self, prompts):
        return ["A"] * len(prompts)

    def loglikelihoods(self, contexts, continuations):
        return [0.0 if c == self.best else -10.0 for c in continuations]


class FakeSingleLoglik(FakeLoglik):
    def __init__(self, best=" A"):
        self.best = best

    def loglikelihood(self, context, continuation):
        return 0.0 if continuation == self.best else -10.0

    @property
    def loglikelihoods(self):
        return None


def test_namespace_exports():
    assert ox.evals.MultipleChoiceTask is MultipleChoiceTask
    assert ox.evals.pass_at_k is pass_at_k
    assert ox.evals.run_suite is run_suite
    assert "evals" in ox.__all__


def test_no_top_level_shadow_of_metrics_accuracy():
    import oxideon.metrics as metrics

    assert ox.accuracy is metrics.accuracy


def test_normalize_answer():
    assert normalize_answer("The Quick, brown fox!") == "quick brown fox"
    assert normalize_answer("  A   B ") == "b"


def test_exact_match():
    assert exact_match(["Paris", "x"], ["paris", "y"]) == 0.5
    assert exact_match(["a"], ["a"]) == 1.0
    with pytest.raises(ValueError):
        exact_match(["a"], ["a", "b"])
    with pytest.raises(ValueError):
        exact_match([], [])


def test_token_f1():
    assert token_f1(["a b c"], ["a b c"]) == 1.0
    assert token_f1(["x b c"], ["x b d"]) == pytest.approx(2 / 3)
    assert token_f1(["x"], ["y"]) == 0.0
    assert token_f1([""], [""]) == 1.0
    assert token_f1(["x"], [""]) == 0.0
    with pytest.raises(ValueError):
        token_f1(["a"], [])


def test_accuracy():
    assert accuracy([1, 2, 3], [1, 0, 3]) == pytest.approx(2 / 3)
    with pytest.raises(ValueError):
        accuracy([], [])


def test_pass_at_k():
    assert pass_at_k(10, 7, 1) == pytest.approx(0.7)
    assert pass_at_k(10, 7, 2) == pytest.approx(1 - 3 / 45)
    assert pass_at_k(5, 5, 3) == 1.0
    assert pass_at_k(5, 0, 2) == 0.0
    with pytest.raises(ValueError):
        pass_at_k(2, 1, 3)
    with pytest.raises(ValueError):
        pass_at_k(2, 3, 1)
    with pytest.raises(ValueError):
        pass_at_k(0, 0, 1)


def test_math_equal():
    assert math_equal("x = \\boxed{42}", "42")
    assert math_equal("So the answer is 12 apples", "12")
    assert math_equal("1,000", "1000")
    assert math_equal("0.333", "0.3330001")
    assert math_equal("yes", "yes")
    assert not math_equal("13", "12")
    assert not math_equal("no number here", "12")


def test_math_equal_nonfinite_never_matches_finite():
    # rel_tol * max(1, |a|, |b|) is infinite when either side is infinite,
    # so a non-finite candidate matched every finite reference pre-fix.
    assert not math_equal("\\boxed{inf}", "123")
    assert not math_equal("\\boxed{1e999}", "5")
    assert not math_equal("\\boxed{5}", "inf")
    assert math_equal("\\boxed{inf}", "inf")


def test_rouge_l():
    assert rouge_l("the cat sat", "the cat sat") == 1.0
    assert rouge_l("a b c", "a x c") == pytest.approx(2 / 3)
    assert rouge_l("a b", "c d") == 0.0
    assert rouge_l("", "") == 1.0
    assert rouge_l("", "a") == 0.0


def test_corpus_bleu():
    assert corpus_bleu(["the cat sat"], [["the cat sat"]]) == pytest.approx(1.0)
    assert corpus_bleu([""], [["nonempty"]]) == 0.0
    assert corpus_bleu(["hello world"], [["goodbye moon"]]) == 0.0
    assert corpus_bleu(["the cat sat"], [["the cat sat on the mat"]]) == pytest.approx(
        math.exp(-1.0)
    )
    with pytest.raises(ValueError):
        corpus_bleu(["a"], [["a"], ["b"]])
    with pytest.raises(ValueError):
        corpus_bleu(["a"], [[]])


def test_perplexity_from_nll():
    assert perplexity_from_nll(0.0) == 1.0
    assert perplexity_from_nll(math.log(2.0)) == pytest.approx(2.0)
    with pytest.raises(ValueError):
        perplexity_from_nll(float("inf"))
    with pytest.raises(ValueError):
        perplexity_from_nll(1000.0)


def test_choice_item_validation():
    with pytest.raises(ValueError):
        ChoiceItem("p", ["only"], 0)
    with pytest.raises(ValueError):
        ChoiceItem("p", ["a", "b"], 2)
    with pytest.raises(ValueError):
        MultipleChoiceTask("t", [])
    items = [ChoiceItem("p", ["a", "b"], 0), ChoiceItem("q", ["a", "b", "c"], 0)]
    with pytest.raises(ValueError):
        MultipleChoiceTask("t", items)
    wide = ChoiceItem("p", [str(i) for i in range(27)], 0)
    with pytest.raises(ValueError):
        MultipleChoiceTask("t", [wide])


def test_multiple_choice_loglik_paths():
    task = MultipleChoiceTask("demo", demo_choices(4))
    assert task.evaluate(FakeLoglik(" A")).metrics["accuracy"] == 1.0
    assert task.evaluate(FakeLoglik(" B")).metrics["accuracy"] == 0.0
    single = FakeSingleLoglik(" A")
    assert task.evaluate(single).metrics["accuracy"] == 1.0
    assert task.evaluate(FakeLoglik(" A"), limit=2).n_items == 2
    with pytest.raises(ValueError):
        task.evaluate(FakeLoglik(" A"), limit=0)


def test_multiple_choice_nan_loglikelihood_is_valueerror():
    # argmax over NaN scores is position-dependent (a NaN first score
    # always wins; a NaN answer score always loses), silently corrupting
    # accuracy. -inf stays legal: an impossible continuation.
    class NanLoglik:
        def loglikelihoods(self, contexts, continuations):
            return [0.0 if i == 0 else float("nan") for i, _ in enumerate(continuations)]

    task = MultipleChoiceTask("demo", demo_choices(2))
    with pytest.raises(ValueError):
        task.evaluate(NanLoglik())

    class NegInfLoglik:
        def loglikelihoods(self, contexts, continuations):
            return [float("-inf") if c == " A" else -1.0 for c in continuations]

    assert task.evaluate(NegInfLoglik()).metrics["accuracy"] == 0.0


def test_multiple_choice_generate_path():
    task = MultipleChoiceTask("demo", demo_choices(4))
    assert task.evaluate(FakeSeq(["A"] * 4)).metrics["accuracy"] == 1.0
    assert task.evaluate(FakeSeq(["B"] * 4)).metrics["accuracy"] == 0.0
    assert task.evaluate(FakeSeq(["???"] * 4)).metrics["accuracy"] == 0.0
    rendered = task.render(0)
    assert "A." in rendered and "Answer:" in rendered


def test_math_task():
    task = MathTask("gsm", demo_math())
    assert len(task) == 4
    good = FakeSeq(["12", "150", "8", "72"])
    assert task.evaluate(good).metrics["accuracy"] == 1.0
    bad = FakeSeq(["0", "0", "0", "0"])
    assert task.evaluate(bad).metrics["accuracy"] == 0.0
    assert task.evaluate(good, limit=2).n_items == 2
    with pytest.raises(ValueError):
        MathTask("empty", [])


def test_run_untrusted_tests():
    assert run_untrusted_tests("x = 1\n", "assert x == 1\n")
    assert not run_untrusted_tests("x = 1\n", "assert x == 2\n")
    assert not run_untrusted_tests("def broken(:\n", "")
    assert not run_untrusted_tests("while True:\n    pass\n", "", timeout=2)
    with pytest.raises(ValueError):
        run_untrusted_tests("x = 1", "assert True", timeout=0)


def test_untrusted_file_writes_stay_contained(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert run_untrusted_tests(
        "open('pwned.txt', 'w').write('x')\n", "assert True\n"
    )
    assert not (tmp_path / "pwned.txt").exists()


def test_code_task():
    task = CodeTask("humaneval", demo_code())
    prompts = [item.prompt for item in demo_code()]
    assert task.evaluate(FakeSeq(prompts)).metrics["pass@1"] == 1.0
    bad = FakeSeq(["def add(a, b):\n    return a - b\n", prompts[1]])
    assert task.evaluate(bad).metrics["pass@1"] == 0.5
    mixed = FakeSeq([prompts[0], "boom(", prompts[1], "boom("])
    result = task.evaluate(mixed, num_samples=2)
    assert result.metrics["pass@1"] == pytest.approx(0.5)
    assert result.metrics["pass@2"] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        CodeTask("empty", [])


def test_make_haystack():
    ctx = make_haystack("NEEDLE HERE", 200, 0.5, seed=1)
    assert "NEEDLE HERE" in ctx
    assert len(ctx.split()) == 200 + 2
    assert make_haystack("N", 50, 0.0).startswith("N ")
    assert make_haystack("N", 50, 1.0).endswith(" N")
    with pytest.raises(ValueError):
        make_haystack("N", 50, 1.5)
    with pytest.raises(ValueError):
        make_haystack("N", 0, 0.5)


def test_needle_task():
    task = NeedleTask(lengths=[64, 128], depths=[0.0, 1.0])
    assert len(task) == 4
    assert "7319" in task.render(0)
    assert task.evaluate(FakeSeq(["the answer is 7319"] * 4)).metrics["accuracy"] == 1.0
    assert task.evaluate(FakeSeq(["no idea"] * 4)).metrics["accuracy"] == 0.0
    assert task.evaluate(FakeSeq(["7319"] * 4), limit=1).n_items == 1
    with pytest.raises(ValueError):
        NeedleTask(lengths=[], depths=[0.0])
    with pytest.raises(ValueError):
        NeedleTask(lengths=[64], depths=[2.0])


def test_run_suite_and_report(tmp_path):
    tasks = [
        MultipleChoiceTask("mc", demo_choices(2)),
        MathTask("math", demo_math()[:2]),
    ]
    report = run_suite(FakeSeq(["A", "A", "12", "150"]), tasks, model_name="fake")
    assert isinstance(report, EvalReport)
    assert [r.name for r in report.results] == ["mc", "math"]
    assert report.to_dict()["model_name"] == "fake"
    text = report.summary()
    assert "mc" in text and "math" in text
    out = report.save(str(tmp_path / "r.json"))
    back = EvalReport.load(out)
    assert back.model_name == "fake"
    assert back.results[0].metrics == report.results[0].metrics
    with pytest.raises(ValueError):
        run_suite(FakeSeq([]), [])
    with pytest.raises(TypeError):
        run_suite(FakeSeq([]), [object()])

    class BadTask:
        name = "bad"

        def evaluate(self, model, limit=None):
            return {"not": "a result"}

    with pytest.raises(TypeError):
        run_suite(FakeSeq([]), [BadTask()])


def test_run_suite_task_kwargs_and_limit():
    task = CodeTask("code", demo_code())
    both = [item.prompt for item in demo_code()]
    report = run_suite(
        FakeSeq([both[0], both[0], both[1], both[1]]),
        [task],
        task_kwargs={"code": {"num_samples": 2, "timeout": 5}},
    )
    assert report.results[0].metrics["pass@1"] == pytest.approx(1.0)
    mc = MultipleChoiceTask("mc", demo_choices(4))
    report = run_suite(FakeSeq(["A"] * 2), [mc], limit=2)
    assert report.results[0].n_items == 2


def test_load_jsonl(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"a": 1}\n\n{"a": 2}\n', encoding="utf-8")
    assert load_jsonl(str(path)) == [{"a": 1}, {"a": 2}]
    bad = tmp_path / "bad.jsonl"
    bad.write_text("[1, 2]\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_jsonl(str(bad))
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_jsonl(str(empty))


def test_model_without_generate_raises():
    task = MathTask("m", demo_math()[:1])
    with pytest.raises(TypeError):
        task.evaluate(object())


def test_rag_context_precision():
    from oxideon.evals import rag_context_precision

    assert rag_context_precision([True, False, True], k=2) == 0.5
    assert rag_context_precision([True, False, True]) == pytest.approx(2 / 3)
    with pytest.raises(ValueError):
        rag_context_precision([], k=1)
    with pytest.raises(ValueError):
        rag_context_precision([True], k=0)
    with pytest.raises(ValueError):
        rag_context_precision([True], k=2)
    with pytest.raises(ValueError):
        rag_context_precision([True, 2])


def test_rag_context_recall_and_faithfulness():
    from oxideon.evals import rag_context_recall, rag_faithfulness

    assert rag_context_recall([True, True, False]) == pytest.approx(2 / 3)
    assert rag_faithfulness([True, True, False]) == pytest.approx(2 / 3)
    assert rag_faithfulness([1, 0, 1]) == pytest.approx(2 / 3)
    with pytest.raises(ValueError):
        rag_context_recall([])
    with pytest.raises(ValueError):
        rag_faithfulness([])


def test_rag_answer_relevancy():
    from oxideon.evals import rag_answer_relevancy

    assert rag_answer_relevancy([1.0, 0.5]) == 0.75
    with pytest.raises(ValueError):
        rag_answer_relevancy([])
    with pytest.raises(ValueError):
        rag_answer_relevancy([0.5, float("nan")])
