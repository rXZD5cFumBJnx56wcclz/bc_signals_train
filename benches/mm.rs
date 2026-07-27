mod prelude;
use prelude::*;

use bc_signals_train::mm::*;

static SIGNAL: LazyLock<fn() -> MM> =
    LazyLock::new(|| || MM::new(0, 0, 2, 3, 0.0001, 0.01, 0.0, 1.0, 2.0));

fn mm_with_bf_1(c: &mut Criterion) {
    let s = SIGNAL();
    let src = &SRC[SRC.len() - 1];
    s.init_bf(&*SRC);
    c.bench_function("mm_with_bf", |b| {
        b.iter(|| s.signal_with_bf(black_box(src)))
    });
}

fn mm_signal_1(c: &mut Criterion) {
    let s = SIGNAL();
    let src = &*SRC;
    c.bench_function("mm_signal_1", |b| b.iter(|| s.signal(black_box(&src))));
}

fn mm_coll_1(c: &mut Criterion) {
    let s = SIGNAL();
    let src = &*SRC;
    c.bench_function("mm_coll_1", |b| {
        b.iter(|| s.signal_coll::<Vec<_>>(black_box(&src)))
    });
}

criterion_group!(benches, mm_with_bf_1, mm_signal_1, mm_coll_1);
criterion_main!(benches);
