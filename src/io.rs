use std::io::BufRead;

use rsomics_common::{Result, RsomicsError};

pub struct Embedding {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
}

/// Read an N×2 whitespace/TSV embedding. A non-numeric first line is treated
/// as a header and skipped. Extra columns past the first two are ignored.
pub fn read_embedding<R: BufRead>(r: R) -> Result<Embedding> {
    let mut x = Vec::new();
    let mut y = Vec::new();
    for (lineno, line) in r.lines().enumerate() {
        let line = line.map_err(RsomicsError::Io)?;
        let t = line.trim();
        if t.is_empty() {
            continue;
        }
        let mut it = t.split_whitespace();
        let (Some(a), Some(b)) = (it.next(), it.next()) else {
            return Err(RsomicsError::InvalidInput(format!(
                "embedding line {} has fewer than 2 columns",
                lineno + 1
            )));
        };
        match (a.parse::<f64>(), b.parse::<f64>()) {
            (Ok(a), Ok(b)) => {
                x.push(a);
                y.push(b);
            }
            _ if lineno == 0 => {} // header
            _ => {
                return Err(RsomicsError::InvalidInput(format!(
                    "embedding line {}: non-numeric coordinate",
                    lineno + 1
                )));
            }
        }
    }
    if x.is_empty() {
        return Err(RsomicsError::InvalidInput("empty embedding".into()));
    }
    Ok(Embedding { x, y })
}

/// One label per line, in cell order. A label `t` that fails to parse as a
/// float on the first line is allowed (labels are categorical strings).
pub fn read_groups<R: BufRead>(r: R, n: usize) -> Result<Vec<String>> {
    let mut g = Vec::with_capacity(n);
    for line in r.lines() {
        let line = line.map_err(RsomicsError::Io)?;
        let t = line.trim();
        if t.is_empty() {
            continue;
        }
        g.push(t.to_string());
    }
    if g.len() != n {
        return Err(RsomicsError::InvalidInput(format!(
            "group labels ({}) do not match embedding rows ({n})",
            g.len()
        )));
    }
    Ok(g)
}
