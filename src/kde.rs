use rayon::prelude::*;

/// 2D Gaussian KDE evaluated at the same points it is fit on, then min-max
/// scaled to [0,1]. Mirrors scanpy's `_calc_density`: `gaussian_kde(xy)(xy)`
/// with scipy's Scott-rule bandwidth, followed by `(z-min)/(max-min)`.
///
/// scipy whitens by the lower Cholesky of the bandwidth-scaled data
/// covariance and accumulates `exp(-||Δ||²/2) * norm` per pair; we reproduce
/// that arithmetic so values match to ~1e-12. A singular covariance (fewer
/// than 2 points, or points lying on a line) is the scipy `LinAlgError` case
/// and is reported as an error rather than silently emitting NaN.
pub fn density(x: &[f64], y: &[f64]) -> Result<Vec<f64>, KdeError> {
    let n = x.len();
    debug_assert_eq!(n, y.len());
    if n < 2 {
        return Err(KdeError::TooFew(n));
    }

    let (mx, my) = mean2(x, y);
    let (cxx, cxy, cyy) = cov2(x, y, mx, my);

    // scipy Cholesky-factors the unscaled data covariance, then scales the
    // factor by `factor` (= chol(data_cov) * factor). Doing it in this order —
    // rather than chol(data_cov * factor²) — reproduces scipy's exact roundoff,
    // which decides the singular/near-singular boundary (LAPACK dpotrf fails
    // iff a leading minor is ≤ 0).
    let factor = (n as f64).powf(-1.0 / 6.0);
    let (c00, c10, c11) = chol_lower(cxx, cxy, cyy).ok_or(KdeError::Singular)?;
    let (l00, l10, l11) = (c00 * factor, c10 * factor, c11 * factor);

    let inv00 = 1.0 / l00;
    let inv11 = 1.0 / l11;
    let off = -l10 * inv00 * inv11;

    let mut wx = vec![0.0f64; n];
    let mut wy = vec![0.0f64; n];
    for i in 0..n {
        wx[i] = x[i] * inv00;
        wy[i] = x[i] * off + y[i] * inv11;
    }

    let norm = (1.0 / (2.0 * std::f64::consts::PI)) / (l00 * l11);

    let z: Vec<f64> = (0..n)
        .into_par_iter()
        .map(|j| {
            let pjx = wx[j];
            let pjy = wy[j];
            let mut acc = 0.0;
            for i in 0..n {
                let dx = wx[i] - pjx;
                let dy = wy[i] - pjy;
                acc += (-(dx * dx + dy * dy) * 0.5).exp();
            }
            acc * norm
        })
        .collect();

    let zmin = z.iter().copied().fold(f64::INFINITY, f64::min);
    let zmax = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let span = zmax - zmin;
    Ok(z.into_iter().map(|v| (v - zmin) / span).collect())
}

#[derive(Debug)]
pub enum KdeError {
    TooFew(usize),
    Singular,
}

fn mean2(x: &[f64], y: &[f64]) -> (f64, f64) {
    let n = x.len() as f64;
    let (mut sx, mut sy) = (0.0, 0.0);
    for i in 0..x.len() {
        sx += x[i];
        sy += y[i];
    }
    (sx / n, sy / n)
}

/// Sample covariance with ddof=1, matching numpy.cov(bias=False).
fn cov2(x: &[f64], y: &[f64], mx: f64, my: f64) -> (f64, f64, f64) {
    let (mut sxx, mut sxy, mut syy) = (0.0, 0.0, 0.0);
    for i in 0..x.len() {
        let dx = x[i] - mx;
        let dy = y[i] - my;
        sxx += dx * dx;
        sxy += dx * dy;
        syy += dy * dy;
    }
    let d = (x.len() - 1) as f64;
    (sxx / d, sxy / d, syy / d)
}

fn chol_lower(a: f64, b: f64, c: f64) -> Option<(f64, f64, f64)> {
    if a <= 0.0 {
        return None;
    }
    let l00 = a.sqrt();
    let l10 = b / l00;
    let rem = c - l10 * l10;
    if rem <= 0.0 {
        return None;
    }
    Some((l00, l10, rem.sqrt()))
}
