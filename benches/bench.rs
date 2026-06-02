use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use rsomics_sc_embedding_density::{Config, Embedding, run};

fn synth(n: usize) -> Embedding {
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut s = 0x2545_F491_4F6C_DD1Du64;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64
    };
    for _ in 0..n {
        x.push(next() * 20.0 - 10.0);
        y.push(next() * 20.0 - 10.0);
    }
    Embedding { x, y }
}

fn bench(c: &mut Criterion) {
    let emb = synth(3000);
    c.bench_function("density_3000", |b| {
        b.iter(|| {
            let mut out = Vec::new();
            run(black_box(&emb), None, &mut out, &Config { precision: 12 }).unwrap();
            black_box(out);
        })
    });
}

criterion_group!(benches, bench);
criterion_main!(benches);
