use std::collections::BTreeMap;
use std::io::Write;

use rsomics_common::{Result, RsomicsError};

mod io;
mod kde;

pub use io::{Embedding, read_embedding, read_groups};

pub struct Config {
    pub precision: usize,
}

/// Compute the per-cell embedding density. With no groups every cell shares
/// one KDE; with groups each category is KDE'd and min-max-scaled on its own,
/// then written back at the cell's original row. Output is a one-column TSV
/// (`<basis>_density`) in input order.
pub fn run<W: Write>(
    emb: &Embedding,
    groups: Option<&[String]>,
    out: &mut W,
    cfg: &Config,
) -> Result<()> {
    let n = emb.x.len();
    let mut dens = vec![0.0f64; n];

    match groups {
        None => {
            dens = kde::density(&emb.x, &emb.y).map_err(map_err)?;
        }
        Some(labels) => {
            let mut by_cat: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
            for (i, g) in labels.iter().enumerate() {
                by_cat.entry(g.as_str()).or_default().push(i);
            }
            for (cat, idx) in &by_cat {
                let cx: Vec<f64> = idx.iter().map(|&i| emb.x[i]).collect();
                let cy: Vec<f64> = idx.iter().map(|&i| emb.y[i]).collect();
                let d = kde::density(&cx, &cy).map_err(|e| match e {
                    kde::KdeError::TooFew(k) => RsomicsError::InvalidInput(format!(
                        "group {cat:?} has {k} cell(s); KDE needs at least 2"
                    )),
                    kde::KdeError::Singular => RsomicsError::InvalidInput(format!(
                        "group {cat:?}: embedding covariance is singular (collinear cells)"
                    )),
                })?;
                for (slot, &i) in idx.iter().enumerate() {
                    dens[i] = d[slot];
                }
            }
        }
    }

    writeln!(out, "density").map_err(RsomicsError::Io)?;
    for &v in &dens {
        if v.is_nan() {
            writeln!(out, "nan").map_err(RsomicsError::Io)?;
        } else {
            writeln!(out, "{v:.*}", cfg.precision).map_err(RsomicsError::Io)?;
        }
    }
    Ok(())
}

fn map_err(e: kde::KdeError) -> RsomicsError {
    match e {
        kde::KdeError::TooFew(k) => {
            RsomicsError::InvalidInput(format!("{k} cell(s); KDE needs at least 2"))
        }
        kde::KdeError::Singular => {
            RsomicsError::InvalidInput("embedding covariance is singular (collinear cells)".into())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_oracle_tiny() {
        let x = [0.1, 0.5, 0.9, 0.2, 0.7, 0.3, 0.8, 0.15];
        let y = [0.2, 0.6, 0.1, 0.8, 0.5, 0.35, 0.9, 0.05];
        let emb = Embedding {
            x: x.to_vec(),
            y: y.to_vec(),
        };
        let mut buf = Vec::new();
        run(&emb, None, &mut buf, &Config { precision: 6 }).unwrap();
        let got: Vec<f64> = String::from_utf8(buf)
            .unwrap()
            .lines()
            .skip(1)
            .map(|l| l.parse().unwrap())
            .collect();
        let want = [
            0.850827, 0.868663, 0.0, 0.154594, 0.600892, 1.0, 0.231796, 0.6725,
        ];
        for (g, w) in got.iter().zip(want.iter()) {
            assert!((g - w).abs() < 1e-6, "{g} vs {w}");
        }
    }

    #[test]
    fn n2_is_nan() {
        let emb = Embedding {
            x: vec![0.1, 0.9],
            y: vec![0.2, 0.8],
        };
        let mut buf = Vec::new();
        run(&emb, None, &mut buf, &Config { precision: 6 }).unwrap();
        let s = String::from_utf8(buf).unwrap();
        assert!(s.lines().skip(1).all(|l| l == "nan"), "{s}");
    }

    #[test]
    fn collinear_errors() {
        let emb = Embedding {
            x: vec![0.0, 1.0, 2.0],
            y: vec![0.0, 1.0, 2.0],
        };
        let mut buf = Vec::new();
        assert!(run(&emb, None, &mut buf, &Config { precision: 6 }).is_err());
    }
}
