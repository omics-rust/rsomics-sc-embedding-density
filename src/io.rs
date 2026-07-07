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
                // `inf`/`nan` parse as valid f64 but poison the covariance into a
                // non-finite Cholesky; scipy raises there, so we reject up front.
                if !a.is_finite() || !b.is_finite() {
                    return Err(RsomicsError::InvalidInput(format!(
                        "embedding line {}: non-finite coordinate",
                        lineno + 1
                    )));
                }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn read(s: &str) -> Result<Embedding> {
        read_embedding(std::io::Cursor::new(s))
    }

    #[test]
    fn rejects_inf_and_nan() {
        for bad in ["inf\t0.5", "-inf 0.5", "0.5 nan", "0.5\tInfinity"] {
            let src = format!("0.1 0.2\n0.3 0.4\n{bad}\n");
            let msg = match read(&src) {
                Ok(_) => panic!("{bad}: accepted a non-finite coordinate"),
                Err(e) => format!("{e}"),
            };
            assert!(msg.contains("non-finite"), "{bad}: {msg}");
        }
    }

    #[test]
    fn accepts_finite() {
        let emb = read("0.1 0.2\n0.3 0.4\n").unwrap();
        assert_eq!(emb.x, vec![0.1, 0.3]);
        assert_eq!(emb.y, vec![0.2, 0.4]);
    }
}
