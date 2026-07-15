use bc_utils_lg::statics::prices::SRC;
use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use std::sync::LazyLock;

use bc_signals_train::mm::*;
use bc_signals_train::prelude::*;

static SIGNALS: LazyLock<Vec<Vec<f64>>> = LazyLock::new(|| Default::default());
static SIGNAL: LazyLock<MM> = LazyLock::new(|| MM::new(0, 0, 2, 3, 0.0001, 0.01, 0.0, 1.0, 2.0));

fn mm_with_bf_1(c: &mut Criterion) {
    let s = &*SIGNAL;
    let src = &SRC[SRC.len() - 1];
    let signals = Default::default();
    let bf = SIGNAL.bf(&*SRC, &*SIGNALS);
    c.bench_function("mm_with_bf", |b| {
        b.iter(|| {
            s.signal_with_bf(
                black_box(src),
                black_box(signals),
                black_box(&bf),
                black_box(0),
            )
        })
    });
}

fn mm_signal_1(c: &mut Criterion) {
    let s = &*SIGNAL;
    let src = &*SRC;
    let signals = &*SIGNALS;
    c.bench_function("mm_signal_1", |b| {
        b.iter(|| s.signal(black_box(&src), black_box(&signals)))
    });
}

fn mm_coll_1(c: &mut Criterion) {
    let s = &*SIGNAL;
    let src = &*SRC;
    let signals = &*SIGNALS;
    c.bench_function("mm_coll_1", |b| {
        b.iter(|| s.signal_coll::<Vec<_>>(black_box(&src), black_box(&signals)))
    });
}

criterion_group!(benches, mm_with_bf_1, mm_signal_1, mm_coll_1);
criterion_main!(benches);
