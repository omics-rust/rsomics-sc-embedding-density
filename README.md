# rsomics-sc-embedding-density

Per-group Gaussian-KDE cell density on a 2D embedding, min-max normalized to
`[0,1]`. Drop-in equivalent of scanpy `sc.tl.embedding_density`: for each
group it fits a 2D `scipy.stats.gaussian_kde` (Scott's-rule bandwidth) over the
cells' embedding coordinates, evaluates it at every cell, and scales the
per-group densities to `[0,1]`.

Unlike scanpy, the embedding is supplied directly — no neighbors/UMAP build is
needed. Feed the first two components of any embedding (`X_umap`, `X_tsne`,
`X_pca`, …).

## Usage

```
# whole-embedding density (one KDE over all cells)
rsomics-sc-embedding-density --embedding umap.tsv -o density.tsv

# per-group density (a KDE per category, each scaled to [0,1] on its own)
rsomics-sc-embedding-density --embedding umap.tsv --groups phase.tsv -o density.tsv
```

- `--embedding` — N×2 whitespace/TSV of embedding coordinates. A non-numeric
  first line is treated as a header.
- `--groups` — optional N-length file of per-cell labels, one per line, in cell
  order. Without it, all cells share one KDE.
- Output — a one-column TSV (`density`) in input cell order.

A group with fewer than 2 cells, or whose cells are collinear, gives a singular
covariance; scipy raises there and so do we (fail loud). A 2-cell group is
symmetric, so its min and max densities coincide and the normalized value is
`nan` — exactly scanpy's behaviour.

## Performance

Single-threaded `>1.0×` vs scanpy `sc.tl.embedding_density` on the same
embedding; the O(N²) per-group kernel sum also scales across cores with rayon.
See `.autopilot/state/perf-sc-embedding-density-*.md` in the control-plane repo
for the recorded provenance.

## Origin

This crate reimplements scanpy `sc.tl.embedding_density` and its underlying
`scipy.stats.gaussian_kde`, both BSD-3-Clause. Their source was read and cited:
the Scott-rule factor `n^(-1/(d+4))`, the ddof=1 data covariance, the
bandwidth-scaled lower-Cholesky whitening, and the `(2π)^(-d/2)/∏Lᵢᵢ` kernel
normalization are reproduced so densities match to ~1e-12.

- scanpy: Wolf, Angerer & Theis, *Genome Biology* 2018, doi:10.1186/s13059-017-1382-0
- scipy: Virtanen et al., *Nature Methods* 2020, doi:10.1038/s41592-019-0686-2
- Scott's Rule: Scott, *Multivariate Density Estimation*, Wiley 1992

License: MIT OR Apache-2.0.
Upstream credit: scanpy (BSD-3-Clause), scipy (BSD-3-Clause).
