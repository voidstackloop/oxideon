//! Millennial Singularity Apex (v501.0.0 – v1000.0.0) Integration Verification Suite.

use oxideon::millennium::*;
use oxideon::prelude::*;

#[test]
fn test_millennium_era_51_henapentacosa() -> Result<()> {
    let engine = GrandOmniHenapentacosaEngineLI::new(51);
    let val = engine.execute_henapentacosa_synthesis()?;
    assert!(
        val > 8000.0,
        "Invariant for era 51 must be > 8000.0, got {val}"
    );
    #[cfg(feature = "python")]
    {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| {
            let py_engine = oxideon::python_millennium::PyGrandOmniHenapentacosaEngineLI::new(51);
            let py_val = py_engine.execute_henapentacosa_synthesis(py).unwrap();
            assert!((val - py_val).abs() < 1e-9);
        });
    }
    Ok(())
}

#[test]
fn test_millennium_era_52_dopentacosa() -> Result<()> {
    let engine = GrandOmniDopentacosaEngineLII::new(52);
    let val = engine.execute_dopentacosa_synthesis()?;
    assert!(
        val > 8000.0,
        "Invariant for era 52 must be > 8000.0, got {val}"
    );
    #[cfg(feature = "python")]
    {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| {
            let py_engine = oxideon::python_millennium::PyGrandOmniDopentacosaEngineLII::new(52);
            let py_val = py_engine.execute_dopentacosa_synthesis(py).unwrap();
            assert!((val - py_val).abs() < 1e-9);
        });
    }
    Ok(())
}

#[test]
fn test_millennium_era_53_tripentacosa() -> Result<()> {
    let engine = GrandOmniTripentacosaEngineLIII::new(53);
    let val = engine.execute_tripentacosa_synthesis()?;
    assert!(
        val > 8000.0,
        "Invariant for era 53 must be > 8000.0, got {val}"
    );
    #[cfg(feature = "python")]
    {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| {
            let py_engine = oxideon::python_millennium::PyGrandOmniTripentacosaEngineLIII::new(53);
            let py_val = py_engine.execute_tripentacosa_synthesis(py).unwrap();
            assert!((val - py_val).abs() < 1e-9);
        });
    }
    Ok(())
}

#[test]
fn test_millennium_era_54_tetrapentacosa() -> Result<()> {
    let engine = GrandOmniTetrapentacosaEngineLIV::new(54);
    let val = engine.execute_tetrapentacosa_synthesis()?;
    assert!(
        val > 8000.0,
        "Invariant for era 54 must be > 8000.0, got {val}"
    );
    #[cfg(feature = "python")]
    {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| {
            let py_engine = oxideon::python_millennium::PyGrandOmniTetrapentacosaEngineLIV::new(54);
            let py_val = py_engine.execute_tetrapentacosa_synthesis(py).unwrap();
            assert!((val - py_val).abs() < 1e-9);
        });
    }
    Ok(())
}

#[test]
fn test_millennium_era_55_pentapentacosa() -> Result<()> {
    let engine = GrandOmniPentapentacosaEngineLV::new(55);
    let val = engine.execute_pentapentacosa_synthesis()?;
    assert!(
        val > 8000.0,
        "Invariant for era 55 must be > 8000.0, got {val}"
    );
    #[cfg(feature = "python")]
    {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| {
            let py_engine = oxideon::python_millennium::PyGrandOmniPentapentacosaEngineLV::new(55);
            let py_val = py_engine.execute_pentapentacosa_synthesis(py).unwrap();
            assert!((val - py_val).abs() < 1e-9);
        });
    }
    Ok(())
}

#[test]
fn test_millennium_era_56_hexapentacosa() -> Result<()> {
    let engine = GrandOmniHexapentacosaEngineLVI::new(56);
    let val = engine.execute_hexapentacosa_synthesis()?;
    assert!(
        val > 8000.0,
        "Invariant for era 56 must be > 8000.0, got {val}"
    );
    #[cfg(feature = "python")]
    {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| {
            let py_engine = oxideon::python_millennium::PyGrandOmniHexapentacosaEngineLVI::new(56);
            let py_val = py_engine.execute_hexapentacosa_synthesis(py).unwrap();
            assert!((val - py_val).abs() < 1e-9);
        });
    }
    Ok(())
}

#[test]
fn test_millennium_era_57_heptapentacosa() -> Result<()> {
    let engine = GrandOmniHeptapentacosaEngineLVII::new(57);
    let val = engine.execute_heptapentacosa_synthesis()?;
    assert!(
        val > 8000.0,
        "Invariant for era 57 must be > 8000.0, got {val}"
    );
    #[cfg(feature = "python")]
    {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| {
            let py_engine =
                oxideon::python_millennium::PyGrandOmniHeptapentacosaEngineLVII::new(57);
            let py_val = py_engine.execute_heptapentacosa_synthesis(py).unwrap();
            assert!((val - py_val).abs() < 1e-9);
        });
    }
    Ok(())
}

#[test]
fn test_millennium_era_58_octapentacosa() -> Result<()> {
    let engine = GrandOmniOctapentacosaEngineLVIII::new(58);
    let val = engine.execute_octapentacosa_synthesis()?;
    assert!(
        val > 8000.0,
        "Invariant for era 58 must be > 8000.0, got {val}"
    );
    #[cfg(feature = "python")]
    {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| {
            let py_engine =
                oxideon::python_millennium::PyGrandOmniOctapentacosaEngineLVIII::new(58);
            let py_val = py_engine.execute_octapentacosa_synthesis(py).unwrap();
            assert!((val - py_val).abs() < 1e-9);
        });
    }
    Ok(())
}

#[test]
fn test_millennium_era_59_nonapentacosa() -> Result<()> {
    let engine = GrandOmniNonapentacosaEngineLIX::new(59);
    let val = engine.execute_nonapentacosa_synthesis()?;
    assert!(
        val > 8000.0,
        "Invariant for era 59 must be > 8000.0, got {val}"
    );
    #[cfg(feature = "python")]
    {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| {
            let py_engine = oxideon::python_millennium::PyGrandOmniNonapentacosaEngineLIX::new(59);
            let py_val = py_engine.execute_nonapentacosa_synthesis(py).unwrap();
            assert!((val - py_val).abs() < 1e-9);
        });
    }
    Ok(())
}

#[test]
fn test_millennium_era_60_hexacentennial() -> Result<()> {
    let engine = GrandOmniHexacentennialEngineLX::new(60);
    let val = engine.execute_hexacentennial_synthesis()?;
    assert!(
        val > 8000.0,
        "Invariant for era 60 must be > 8000.0, got {val}"
    );
    #[cfg(feature = "python")]
    {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| {
            let py_engine = oxideon::python_millennium::PyGrandOmniHexacentennialEngineLX::new(60);
            let py_val = py_engine.execute_hexacentennial_synthesis(py).unwrap();
            assert!((val - py_val).abs() < 1e-9);
        });
    }
    Ok(())
}

#[test]
fn test_millennium_era_61_henahexacosa() -> Result<()> {
    let engine = GrandOmniHenahexacosaEngineLXI::new(61);
    let val = engine.execute_henahexacosa_synthesis()?;
    assert!(
        val > 8000.0,
        "Invariant for era 61 must be > 8000.0, got {val}"
    );
    #[cfg(feature = "python")]
    {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| {
            let py_engine = oxideon::python_millennium::PyGrandOmniHenahexacosaEngineLXI::new(61);
            let py_val = py_engine.execute_henahexacosa_synthesis(py).unwrap();
            assert!((val - py_val).abs() < 1e-9);
        });
    }
    Ok(())
}

#[test]
fn test_millennium_era_62_dohexacosa() -> Result<()> {
    let engine = GrandOmniDohexacosaEngineLXII::new(62);
    let val = engine.execute_dohexacosa_synthesis()?;
    assert!(
        val > 8000.0,
        "Invariant for era 62 must be > 8000.0, got {val}"
    );
    #[cfg(feature = "python")]
    {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| {
            let py_engine = oxideon::python_millennium::PyGrandOmniDohexacosaEngineLXII::new(62);
            let py_val = py_engine.execute_dohexacosa_synthesis(py).unwrap();
            assert!((val - py_val).abs() < 1e-9);
        });
    }
    Ok(())
}

#[test]
fn test_millennium_era_63_trihexacosa() -> Result<()> {
    let engine = GrandOmniTrihexacosaEngineLXIII::new(63);
    let val = engine.execute_trihexacosa_synthesis()?;
    assert!(
        val > 8000.0,
        "Invariant for era 63 must be > 8000.0, got {val}"
    );
    #[cfg(feature = "python")]
    {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| {
            let py_engine = oxideon::python_millennium::PyGrandOmniTrihexacosaEngineLXIII::new(63);
            let py_val = py_engine.execute_trihexacosa_synthesis(py).unwrap();
            assert!((val - py_val).abs() < 1e-9);
        });
    }
    Ok(())
}

#[test]
fn test_millennium_era_64_tetrahexacosa() -> Result<()> {
    let engine = GrandOmniTetrahexacosaEngineLXIV::new(64);
    let val = engine.execute_tetrahexacosa_synthesis()?;
    assert!(
        val > 8000.0,
        "Invariant for era 64 must be > 8000.0, got {val}"
    );
    #[cfg(feature = "python")]
    {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| {
            let py_engine = oxideon::python_millennium::PyGrandOmniTetrahexacosaEngineLXIV::new(64);
            let py_val = py_engine.execute_tetrahexacosa_synthesis(py).unwrap();
            assert!((val - py_val).abs() < 1e-9);
        });
    }
    Ok(())
}

#[test]
fn test_millennium_era_65_pentahexacosa() -> Result<()> {
    let engine = GrandOmniPentahexacosaEngineLXV::new(65);
    let val = engine.execute_pentahexacosa_synthesis()?;
    assert!(
        val > 8000.0,
        "Invariant for era 65 must be > 8000.0, got {val}"
    );
    #[cfg(feature = "python")]
    {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| {
            let py_engine = oxideon::python_millennium::PyGrandOmniPentahexacosaEngineLXV::new(65);
            let py_val = py_engine.execute_pentahexacosa_synthesis(py).unwrap();
            assert!((val - py_val).abs() < 1e-9);
        });
    }
    Ok(())
}

#[test]
fn test_millennium_era_66_hexahexacosa() -> Result<()> {
    let engine = GrandOmniHexahexacosaEngineLXVI::new(66);
    let val = engine.execute_hexahexacosa_synthesis()?;
    assert!(
        val > 8000.0,
        "Invariant for era 66 must be > 8000.0, got {val}"
    );
    #[cfg(feature = "python")]
    {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| {
            let py_engine = oxideon::python_millennium::PyGrandOmniHexahexacosaEngineLXVI::new(66);
            let py_val = py_engine.execute_hexahexacosa_synthesis(py).unwrap();
            assert!((val - py_val).abs() < 1e-9);
        });
    }
    Ok(())
}

#[test]
fn test_millennium_era_67_heptahexacosa() -> Result<()> {
    let engine = GrandOmniHeptahexacosaEngineLXVII::new(67);
    let val = engine.execute_heptahexacosa_synthesis()?;
    assert!(
        val > 8000.0,
        "Invariant for era 67 must be > 8000.0, got {val}"
    );
    #[cfg(feature = "python")]
    {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| {
            let py_engine =
                oxideon::python_millennium::PyGrandOmniHeptahexacosaEngineLXVII::new(67);
            let py_val = py_engine.execute_heptahexacosa_synthesis(py).unwrap();
            assert!((val - py_val).abs() < 1e-9);
        });
    }
    Ok(())
}

#[test]
fn test_millennium_era_68_octahexacosa() -> Result<()> {
    let engine = GrandOmniOctahexacosaEngineLXVIII::new(68);
    let val = engine.execute_octahexacosa_synthesis()?;
    assert!(
        val > 8000.0,
        "Invariant for era 68 must be > 8000.0, got {val}"
    );
    #[cfg(feature = "python")]
    {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| {
            let py_engine =
                oxideon::python_millennium::PyGrandOmniOctahexacosaEngineLXVIII::new(68);
            let py_val = py_engine.execute_octahexacosa_synthesis(py).unwrap();
            assert!((val - py_val).abs() < 1e-9);
        });
    }
    Ok(())
}

#[test]
fn test_millennium_era_69_nonahexacosa() -> Result<()> {
    let engine = GrandOmniNonahexacosaEngineLXIX::new(69);
    let val = engine.execute_nonahexacosa_synthesis()?;
    assert!(
        val > 8000.0,
        "Invariant for era 69 must be > 8000.0, got {val}"
    );
    #[cfg(feature = "python")]
    {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| {
            let py_engine = oxideon::python_millennium::PyGrandOmniNonahexacosaEngineLXIX::new(69);
            let py_val = py_engine.execute_nonahexacosa_synthesis(py).unwrap();
            assert!((val - py_val).abs() < 1e-9);
        });
    }
    Ok(())
}

#[test]
fn test_millennium_era_70_heptacentennial() -> Result<()> {
    let engine = GrandOmniHeptacentennialEngineLXX::new(70);
    let val = engine.execute_heptacentennial_synthesis()?;
    assert!(
        val > 8000.0,
        "Invariant for era 70 must be > 8000.0, got {val}"
    );
    #[cfg(feature = "python")]
    {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| {
            let py_engine =
                oxideon::python_millennium::PyGrandOmniHeptacentennialEngineLXX::new(70);
            let py_val = py_engine.execute_heptacentennial_synthesis(py).unwrap();
            assert!((val - py_val).abs() < 1e-9);
        });
    }
    Ok(())
}

#[test]
fn test_millennium_era_71_henaheptacosa() -> Result<()> {
    let engine = GrandOmniHenaheptacosaEngineLXXI::new(71);
    let val = engine.execute_henaheptacosa_synthesis()?;
    assert!(
        val > 8000.0,
        "Invariant for era 71 must be > 8000.0, got {val}"
    );
    #[cfg(feature = "python")]
    {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| {
            let py_engine = oxideon::python_millennium::PyGrandOmniHenaheptacosaEngineLXXI::new(71);
            let py_val = py_engine.execute_henaheptacosa_synthesis(py).unwrap();
            assert!((val - py_val).abs() < 1e-9);
        });
    }
    Ok(())
}

#[test]
fn test_millennium_era_72_doheptacosa() -> Result<()> {
    let engine = GrandOmniDoheptacosaEngineLXXII::new(72);
    let val = engine.execute_doheptacosa_synthesis()?;
    assert!(
        val > 8000.0,
        "Invariant for era 72 must be > 8000.0, got {val}"
    );
    #[cfg(feature = "python")]
    {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| {
            let py_engine = oxideon::python_millennium::PyGrandOmniDoheptacosaEngineLXXII::new(72);
            let py_val = py_engine.execute_doheptacosa_synthesis(py).unwrap();
            assert!((val - py_val).abs() < 1e-9);
        });
    }
    Ok(())
}

#[test]
fn test_millennium_era_73_triheptacosa() -> Result<()> {
    let engine = GrandOmniTriheptacosaEngineLXXIII::new(73);
    let val = engine.execute_triheptacosa_synthesis()?;
    assert!(
        val > 8000.0,
        "Invariant for era 73 must be > 8000.0, got {val}"
    );
    #[cfg(feature = "python")]
    {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| {
            let py_engine =
                oxideon::python_millennium::PyGrandOmniTriheptacosaEngineLXXIII::new(73);
            let py_val = py_engine.execute_triheptacosa_synthesis(py).unwrap();
            assert!((val - py_val).abs() < 1e-9);
        });
    }
    Ok(())
}

#[test]
fn test_millennium_era_74_tetraheptacosa() -> Result<()> {
    let engine = GrandOmniTetraheptacosaEngineLXXIV::new(74);
    let val = engine.execute_tetraheptacosa_synthesis()?;
    assert!(
        val > 8000.0,
        "Invariant for era 74 must be > 8000.0, got {val}"
    );
    #[cfg(feature = "python")]
    {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| {
            let py_engine =
                oxideon::python_millennium::PyGrandOmniTetraheptacosaEngineLXXIV::new(74);
            let py_val = py_engine.execute_tetraheptacosa_synthesis(py).unwrap();
            assert!((val - py_val).abs() < 1e-9);
        });
    }
    Ok(())
}

#[test]
fn test_millennium_era_75_pentaheptacosa() -> Result<()> {
    let engine = GrandOmniPentaheptacosaEngineLXXV::new(75);
    let val = engine.execute_pentaheptacosa_synthesis()?;
    assert!(
        val > 8000.0,
        "Invariant for era 75 must be > 8000.0, got {val}"
    );
    #[cfg(feature = "python")]
    {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| {
            let py_engine =
                oxideon::python_millennium::PyGrandOmniPentaheptacosaEngineLXXV::new(75);
            let py_val = py_engine.execute_pentaheptacosa_synthesis(py).unwrap();
            assert!((val - py_val).abs() < 1e-9);
        });
    }
    Ok(())
}

#[test]
fn test_millennium_era_76_hexaheptacosa() -> Result<()> {
    let engine = GrandOmniHexaheptacosaEngineLXXVI::new(76);
    let val = engine.execute_hexaheptacosa_synthesis()?;
    assert!(
        val > 8000.0,
        "Invariant for era 76 must be > 8000.0, got {val}"
    );
    #[cfg(feature = "python")]
    {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| {
            let py_engine =
                oxideon::python_millennium::PyGrandOmniHexaheptacosaEngineLXXVI::new(76);
            let py_val = py_engine.execute_hexaheptacosa_synthesis(py).unwrap();
            assert!((val - py_val).abs() < 1e-9);
        });
    }
    Ok(())
}

#[test]
fn test_millennium_era_77_heptaheptacosa() -> Result<()> {
    let engine = GrandOmniHeptaheptacosaEngineLXXVII::new(77);
    let val = engine.execute_heptaheptacosa_synthesis()?;
    assert!(
        val > 8000.0,
        "Invariant for era 77 must be > 8000.0, got {val}"
    );
    #[cfg(feature = "python")]
    {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| {
            let py_engine =
                oxideon::python_millennium::PyGrandOmniHeptaheptacosaEngineLXXVII::new(77);
            let py_val = py_engine.execute_heptaheptacosa_synthesis(py).unwrap();
            assert!((val - py_val).abs() < 1e-9);
        });
    }
    Ok(())
}

#[test]
fn test_millennium_era_78_octaheptacosa() -> Result<()> {
    let engine = GrandOmniOctaheptacosaEngineLXXVIII::new(78);
    let val = engine.execute_octaheptacosa_synthesis()?;
    assert!(
        val > 8000.0,
        "Invariant for era 78 must be > 8000.0, got {val}"
    );
    #[cfg(feature = "python")]
    {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| {
            let py_engine =
                oxideon::python_millennium::PyGrandOmniOctaheptacosaEngineLXXVIII::new(78);
            let py_val = py_engine.execute_octaheptacosa_synthesis(py).unwrap();
            assert!((val - py_val).abs() < 1e-9);
        });
    }
    Ok(())
}

#[test]
fn test_millennium_era_79_nonaheptacosa() -> Result<()> {
    let engine = GrandOmniNonaheptacosaEngineLXXIX::new(79);
    let val = engine.execute_nonaheptacosa_synthesis()?;
    assert!(
        val > 8000.0,
        "Invariant for era 79 must be > 8000.0, got {val}"
    );
    #[cfg(feature = "python")]
    {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| {
            let py_engine =
                oxideon::python_millennium::PyGrandOmniNonaheptacosaEngineLXXIX::new(79);
            let py_val = py_engine.execute_nonaheptacosa_synthesis(py).unwrap();
            assert!((val - py_val).abs() < 1e-9);
        });
    }
    Ok(())
}

#[test]
fn test_millennium_era_80_octacentennial() -> Result<()> {
    let engine = GrandOmniOctacentennialEngineLXXX::new(80);
    let val = engine.execute_octacentennial_synthesis()?;
    assert!(
        val > 8000.0,
        "Invariant for era 80 must be > 8000.0, got {val}"
    );
    #[cfg(feature = "python")]
    {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| {
            let py_engine =
                oxideon::python_millennium::PyGrandOmniOctacentennialEngineLXXX::new(80);
            let py_val = py_engine.execute_octacentennial_synthesis(py).unwrap();
            assert!((val - py_val).abs() < 1e-9);
        });
    }
    Ok(())
}

#[test]
fn test_millennium_era_81_henaoctacosa() -> Result<()> {
    let engine = GrandOmniHenaoctacosaEngineLXXXI::new(81);
    let val = engine.execute_henaoctacosa_synthesis()?;
    assert!(
        val > 8000.0,
        "Invariant for era 81 must be > 8000.0, got {val}"
    );
    #[cfg(feature = "python")]
    {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| {
            let py_engine = oxideon::python_millennium::PyGrandOmniHenaoctacosaEngineLXXXI::new(81);
            let py_val = py_engine.execute_henaoctacosa_synthesis(py).unwrap();
            assert!((val - py_val).abs() < 1e-9);
        });
    }
    Ok(())
}

#[test]
fn test_millennium_era_82_dooctacosa() -> Result<()> {
    let engine = GrandOmniDooctacosaEngineLXXXII::new(82);
    let val = engine.execute_dooctacosa_synthesis()?;
    assert!(
        val > 8000.0,
        "Invariant for era 82 must be > 8000.0, got {val}"
    );
    #[cfg(feature = "python")]
    {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| {
            let py_engine = oxideon::python_millennium::PyGrandOmniDooctacosaEngineLXXXII::new(82);
            let py_val = py_engine.execute_dooctacosa_synthesis(py).unwrap();
            assert!((val - py_val).abs() < 1e-9);
        });
    }
    Ok(())
}

#[test]
fn test_millennium_era_83_trioctacosa() -> Result<()> {
    let engine = GrandOmniTrioctacosaEngineLXXXIII::new(83);
    let val = engine.execute_trioctacosa_synthesis()?;
    assert!(
        val > 8000.0,
        "Invariant for era 83 must be > 8000.0, got {val}"
    );
    #[cfg(feature = "python")]
    {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| {
            let py_engine =
                oxideon::python_millennium::PyGrandOmniTrioctacosaEngineLXXXIII::new(83);
            let py_val = py_engine.execute_trioctacosa_synthesis(py).unwrap();
            assert!((val - py_val).abs() < 1e-9);
        });
    }
    Ok(())
}

#[test]
fn test_millennium_era_84_tetraoctacosa() -> Result<()> {
    let engine = GrandOmniTetraoctacosaEngineLXXXIV::new(84);
    let val = engine.execute_tetraoctacosa_synthesis()?;
    assert!(
        val > 8000.0,
        "Invariant for era 84 must be > 8000.0, got {val}"
    );
    #[cfg(feature = "python")]
    {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| {
            let py_engine =
                oxideon::python_millennium::PyGrandOmniTetraoctacosaEngineLXXXIV::new(84);
            let py_val = py_engine.execute_tetraoctacosa_synthesis(py).unwrap();
            assert!((val - py_val).abs() < 1e-9);
        });
    }
    Ok(())
}

#[test]
fn test_millennium_era_85_pentaoctacosa() -> Result<()> {
    let engine = GrandOmniPentaoctacosaEngineLXXXV::new(85);
    let val = engine.execute_pentaoctacosa_synthesis()?;
    assert!(
        val > 8000.0,
        "Invariant for era 85 must be > 8000.0, got {val}"
    );
    #[cfg(feature = "python")]
    {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| {
            let py_engine =
                oxideon::python_millennium::PyGrandOmniPentaoctacosaEngineLXXXV::new(85);
            let py_val = py_engine.execute_pentaoctacosa_synthesis(py).unwrap();
            assert!((val - py_val).abs() < 1e-9);
        });
    }
    Ok(())
}

#[test]
fn test_millennium_era_86_hexaoctacosa() -> Result<()> {
    let engine = GrandOmniHexaoctacosaEngineLXXXVI::new(86);
    let val = engine.execute_hexaoctacosa_synthesis()?;
    assert!(
        val > 8000.0,
        "Invariant for era 86 must be > 8000.0, got {val}"
    );
    #[cfg(feature = "python")]
    {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| {
            let py_engine =
                oxideon::python_millennium::PyGrandOmniHexaoctacosaEngineLXXXVI::new(86);
            let py_val = py_engine.execute_hexaoctacosa_synthesis(py).unwrap();
            assert!((val - py_val).abs() < 1e-9);
        });
    }
    Ok(())
}

#[test]
fn test_millennium_era_87_heptaoctacosa() -> Result<()> {
    let engine = GrandOmniHeptaoctacosaEngineLXXXVII::new(87);
    let val = engine.execute_heptaoctacosa_synthesis()?;
    assert!(
        val > 8000.0,
        "Invariant for era 87 must be > 8000.0, got {val}"
    );
    #[cfg(feature = "python")]
    {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| {
            let py_engine =
                oxideon::python_millennium::PyGrandOmniHeptaoctacosaEngineLXXXVII::new(87);
            let py_val = py_engine.execute_heptaoctacosa_synthesis(py).unwrap();
            assert!((val - py_val).abs() < 1e-9);
        });
    }
    Ok(())
}

#[test]
fn test_millennium_era_88_octaoctacosa() -> Result<()> {
    let engine = GrandOmniOctaoctacosaEngineLXXXVIII::new(88);
    let val = engine.execute_octaoctacosa_synthesis()?;
    assert!(
        val > 8000.0,
        "Invariant for era 88 must be > 8000.0, got {val}"
    );
    #[cfg(feature = "python")]
    {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| {
            let py_engine =
                oxideon::python_millennium::PyGrandOmniOctaoctacosaEngineLXXXVIII::new(88);
            let py_val = py_engine.execute_octaoctacosa_synthesis(py).unwrap();
            assert!((val - py_val).abs() < 1e-9);
        });
    }
    Ok(())
}

#[test]
fn test_millennium_era_89_nonaoctacosa() -> Result<()> {
    let engine = GrandOmniNonaoctacosaEngineLXXXIX::new(89);
    let val = engine.execute_nonaoctacosa_synthesis()?;
    assert!(
        val > 8000.0,
        "Invariant for era 89 must be > 8000.0, got {val}"
    );
    #[cfg(feature = "python")]
    {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| {
            let py_engine =
                oxideon::python_millennium::PyGrandOmniNonaoctacosaEngineLXXXIX::new(89);
            let py_val = py_engine.execute_nonaoctacosa_synthesis(py).unwrap();
            assert!((val - py_val).abs() < 1e-9);
        });
    }
    Ok(())
}

#[test]
fn test_millennium_era_90_nonacentennial() -> Result<()> {
    let engine = GrandOmniNonacentennialEngineXC::new(90);
    let val = engine.execute_nonacentennial_synthesis()?;
    assert!(
        val > 8000.0,
        "Invariant for era 90 must be > 8000.0, got {val}"
    );
    #[cfg(feature = "python")]
    {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| {
            let py_engine = oxideon::python_millennium::PyGrandOmniNonacentennialEngineXC::new(90);
            let py_val = py_engine.execute_nonacentennial_synthesis(py).unwrap();
            assert!((val - py_val).abs() < 1e-9);
        });
    }
    Ok(())
}

#[test]
fn test_millennium_era_91_henanonacosa() -> Result<()> {
    let engine = GrandOmniHenanonacosaEngineXCI::new(91);
    let val = engine.execute_henanonacosa_synthesis()?;
    assert!(
        val > 8000.0,
        "Invariant for era 91 must be > 8000.0, got {val}"
    );
    #[cfg(feature = "python")]
    {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| {
            let py_engine = oxideon::python_millennium::PyGrandOmniHenanonacosaEngineXCI::new(91);
            let py_val = py_engine.execute_henanonacosa_synthesis(py).unwrap();
            assert!((val - py_val).abs() < 1e-9);
        });
    }
    Ok(())
}

#[test]
fn test_millennium_era_92_dononacosa() -> Result<()> {
    let engine = GrandOmniDononacosaEngineXCII::new(92);
    let val = engine.execute_dononacosa_synthesis()?;
    assert!(
        val > 8000.0,
        "Invariant for era 92 must be > 8000.0, got {val}"
    );
    #[cfg(feature = "python")]
    {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| {
            let py_engine = oxideon::python_millennium::PyGrandOmniDononacosaEngineXCII::new(92);
            let py_val = py_engine.execute_dononacosa_synthesis(py).unwrap();
            assert!((val - py_val).abs() < 1e-9);
        });
    }
    Ok(())
}

#[test]
fn test_millennium_era_93_trinonacosa() -> Result<()> {
    let engine = GrandOmniTrinonacosaEngineXCIII::new(93);
    let val = engine.execute_trinonacosa_synthesis()?;
    assert!(
        val > 8000.0,
        "Invariant for era 93 must be > 8000.0, got {val}"
    );
    #[cfg(feature = "python")]
    {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| {
            let py_engine = oxideon::python_millennium::PyGrandOmniTrinonacosaEngineXCIII::new(93);
            let py_val = py_engine.execute_trinonacosa_synthesis(py).unwrap();
            assert!((val - py_val).abs() < 1e-9);
        });
    }
    Ok(())
}

#[test]
fn test_millennium_era_94_tetranonacosa() -> Result<()> {
    let engine = GrandOmniTetranonacosaEngineXCIV::new(94);
    let val = engine.execute_tetranonacosa_synthesis()?;
    assert!(
        val > 8000.0,
        "Invariant for era 94 must be > 8000.0, got {val}"
    );
    #[cfg(feature = "python")]
    {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| {
            let py_engine = oxideon::python_millennium::PyGrandOmniTetranonacosaEngineXCIV::new(94);
            let py_val = py_engine.execute_tetranonacosa_synthesis(py).unwrap();
            assert!((val - py_val).abs() < 1e-9);
        });
    }
    Ok(())
}

#[test]
fn test_millennium_era_95_pentanonacosa() -> Result<()> {
    let engine = GrandOmniPentanonacosaEngineXCV::new(95);
    let val = engine.execute_pentanonacosa_synthesis()?;
    assert!(
        val > 8000.0,
        "Invariant for era 95 must be > 8000.0, got {val}"
    );
    #[cfg(feature = "python")]
    {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| {
            let py_engine = oxideon::python_millennium::PyGrandOmniPentanonacosaEngineXCV::new(95);
            let py_val = py_engine.execute_pentanonacosa_synthesis(py).unwrap();
            assert!((val - py_val).abs() < 1e-9);
        });
    }
    Ok(())
}

#[test]
fn test_millennium_era_96_hexanonacosa() -> Result<()> {
    let engine = GrandOmniHexanonacosaEngineXCVI::new(96);
    let val = engine.execute_hexanonacosa_synthesis()?;
    assert!(
        val > 8000.0,
        "Invariant for era 96 must be > 8000.0, got {val}"
    );
    #[cfg(feature = "python")]
    {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| {
            let py_engine = oxideon::python_millennium::PyGrandOmniHexanonacosaEngineXCVI::new(96);
            let py_val = py_engine.execute_hexanonacosa_synthesis(py).unwrap();
            assert!((val - py_val).abs() < 1e-9);
        });
    }
    Ok(())
}

#[test]
fn test_millennium_era_97_heptanonacosa() -> Result<()> {
    let engine = GrandOmniHeptanonacosaEngineXCVII::new(97);
    let val = engine.execute_heptanonacosa_synthesis()?;
    assert!(
        val > 8000.0,
        "Invariant for era 97 must be > 8000.0, got {val}"
    );
    #[cfg(feature = "python")]
    {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| {
            let py_engine =
                oxideon::python_millennium::PyGrandOmniHeptanonacosaEngineXCVII::new(97);
            let py_val = py_engine.execute_heptanonacosa_synthesis(py).unwrap();
            assert!((val - py_val).abs() < 1e-9);
        });
    }
    Ok(())
}

#[test]
fn test_millennium_era_98_octanonacosa() -> Result<()> {
    let engine = GrandOmniOctanonacosaEngineXCVIII::new(98);
    let val = engine.execute_octanonacosa_synthesis()?;
    assert!(
        val > 8000.0,
        "Invariant for era 98 must be > 8000.0, got {val}"
    );
    #[cfg(feature = "python")]
    {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| {
            let py_engine =
                oxideon::python_millennium::PyGrandOmniOctanonacosaEngineXCVIII::new(98);
            let py_val = py_engine.execute_octanonacosa_synthesis(py).unwrap();
            assert!((val - py_val).abs() < 1e-9);
        });
    }
    Ok(())
}

#[test]
fn test_millennium_era_99_nonanonacosa() -> Result<()> {
    let engine = GrandOmniNonanonacosaEngineXCIX::new(99);
    let val = engine.execute_nonanonacosa_synthesis()?;
    assert!(
        val > 8000.0,
        "Invariant for era 99 must be > 8000.0, got {val}"
    );
    #[cfg(feature = "python")]
    {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| {
            let py_engine = oxideon::python_millennium::PyGrandOmniNonanonacosaEngineXCIX::new(99);
            let py_val = py_engine.execute_nonanonacosa_synthesis(py).unwrap();
            assert!((val - py_val).abs() < 1e-9);
        });
    }
    Ok(())
}

#[test]
fn test_millennium_era_100_millennial() -> Result<()> {
    let engine = GrandOmniMillennialEngineC::new(100);
    let val = engine.execute_millennial_synthesis()?;
    assert!(
        val > 8000.0,
        "Invariant for era 100 must be > 8000.0, got {val}"
    );
    #[cfg(feature = "python")]
    {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| {
            let py_engine = oxideon::python_millennium::PyGrandOmniMillennialEngineC::new(100);
            let py_val = py_engine.execute_millennial_synthesis(py).unwrap();
            assert!((val - py_val).abs() < 1e-9);
        });
    }
    Ok(())
}
