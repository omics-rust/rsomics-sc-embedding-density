use std::path::{Path, PathBuf};
use std::process::Command;

const TOL: f64 = 1e-9;

fn golden(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/golden")
        .join(name)
}

fn bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_rsomics-sc-embedding-density"))
}

fn parse(text: &str) -> Vec<f64> {
    text.lines()
        .skip(1)
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            if l.trim().eq_ignore_ascii_case("nan") {
                f64::NAN
            } else {
                l.trim().parse().expect("float")
            }
        })
        .collect()
}

fn assert_close(got: &str, expected: &str, label: &str) {
    let a = parse(got);
    let b = parse(expected);
    assert_eq!(a.len(), b.len(), "{label}: row count differs");
    for (i, (va, vb)) in a.iter().zip(b.iter()).enumerate() {
        if vb.is_nan() {
            assert!(va.is_nan(), "{label}/{i}: expected NaN, got {va}");
        } else {
            let d = (va - vb).abs();
            assert!(
                d <= TOL * (1.0 + vb.abs()),
                "{label}/{i}: {va} vs {vb} (|d|={d})"
            );
        }
    }
}

fn run(groups: Option<&Path>) -> String {
    let mut cmd = Command::new(bin());
    cmd.args([
        "--embedding",
        golden("embedding.tsv").to_str().unwrap(),
        "--precision",
        "15",
    ]);
    if let Some(g) = groups {
        cmd.args(["--groups", g.to_str().unwrap()]);
    }
    let out = cmd.output().expect("run binary");
    assert!(out.status.success(), "binary failed: {:?}", out.status);
    String::from_utf8(out.stdout).unwrap()
}

#[test]
fn whole_embedding_matches_golden() {
    let expected = std::fs::read_to_string(golden("density.expected.tsv")).unwrap();
    assert_close(&run(None), &expected, "whole");
}

#[test]
fn grouped_matches_golden() {
    let expected = std::fs::read_to_string(golden("density_grouped.expected.tsv")).unwrap();
    assert_close(&run(Some(&golden("groups.tsv"))), &expected, "grouped");
}

/// Live differential against scanpy if importable (or via SCANPY_PYTHON).
/// Loud-skip otherwise so CI stays oracle-free.
#[test]
fn live_oracle_diff() {
    let py = std::env::var("SCANPY_PYTHON").unwrap_or_else(|_| "python3".into());
    let probe = Command::new(&py)
        .args(["-c", "import scanpy, numpy, pandas, anndata"])
        .output();
    match probe {
        Ok(o) if o.status.success() => {}
        _ => {
            eprintln!("SKIP live_oracle_diff: scanpy not importable via `{py}`");
            return;
        }
    }

    let scratch = std::env::temp_dir().join("rsomics_sc_embedding_density_oracle");
    std::fs::create_dir_all(&scratch).unwrap();
    let script = scratch.join("oracle.py");
    std::fs::write(&script, ORACLE).unwrap();

    let whole = scratch.join("whole.tsv");
    let grouped = scratch.join("grouped.tsv");
    let o = Command::new(&py)
        .arg(&script)
        .arg(golden("embedding.tsv"))
        .arg(golden("groups.tsv"))
        .arg(&whole)
        .arg(&grouped)
        .output()
        .unwrap();
    assert!(
        o.status.success(),
        "oracle failed: {}",
        String::from_utf8_lossy(&o.stderr)
    );

    assert_close(
        &run(None),
        &std::fs::read_to_string(&whole).unwrap(),
        "live whole",
    );
    assert_close(
        &run(Some(&golden("groups.tsv"))),
        &std::fs::read_to_string(&grouped).unwrap(),
        "live grouped",
    );
}

const ORACLE: &str = r#"
import sys, numpy as np, pandas as pd, anndata as ad, scanpy as sc
emb = np.loadtxt(sys.argv[1])
grp = [l.strip() for l in open(sys.argv[2]) if l.strip()]
n = emb.shape[0]
a = ad.AnnData(np.zeros((n, 1), dtype=np.float32))
a.obsm["X_umap"] = emb
sc.tl.embedding_density(a, basis="umap")
def dump(p, col):
    with open(p, "w") as f:
        f.write("density\n")
        for v in col:
            f.write("nan\n" if np.isnan(v) else f"{repr(float(v))}\n")
dump(sys.argv[3], a.obs["umap_density"].to_numpy())
a.obs["grp"] = pd.Categorical(grp)
sc.tl.embedding_density(a, basis="umap", groupby="grp")
dump(sys.argv[4], a.obs["umap_density_grp"].to_numpy())
"#;
