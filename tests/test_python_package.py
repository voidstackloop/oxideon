"""Standalone verification suite for the installed `oxideon` Python wheel."""

import re

import oxideon as ox


def test_version_and_metadata() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+", ox.__version__), ox.__version__
    assert hasattr(ox, "Tensor")
    assert hasattr(ox, "GrandCentennialSingularityEngineX")


def test_tensor_autograd_and_dl() -> None:
    x = ox.Tensor([2.0, 3.0], [2], requires_grad=True)
    w = ox.Tensor([4.0, 5.0], [2], requires_grad=True)
    y = (x * w).sum()
    y.backward()
    assert x.grad().to_list() == [4.0, 5.0]
    assert w.grad().to_list() == [2.0, 3.0]


def test_dynamic_llm_quantizer_and_centennial_engine() -> None:
    q = ox.DynamicLlmQuantizer()
    w = ox.Tensor([0.1, -0.2, 0.3, 0.4], [2, 2])
    q.add_layer("attn.q_proj", w)
    tel = ox.DynamicTelemetry(45.0, 48.0, 92.0, 3.5, 100)
    plan = q.evaluate_telemetry(tel)
    assert plan is not None

    eng = ox.GrandCentennialSingularityEngineX(12)
    rep = eng.execute_centennial_singularity()
    assert rep.hott_univalence_verified
    assert rep.langlands_deligne_verified
    assert rep.twistor_null_helicity_residual < 1e-13
    assert rep.centennial_singularity_invariant > 100.0


if __name__ == "__main__":
    test_version_and_metadata()
    test_tensor_autograd_and_dl()
    test_dynamic_llm_quantizer_and_centennial_engine()
    print(f"All standalone Python wheel tests passed for oxideon v{ox.__version__}!")
