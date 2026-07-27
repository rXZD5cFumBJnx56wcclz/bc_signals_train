use std::cmp::{Ordering::Equal, min_by_key};

use crate::prelude::*;

#[derive(Clone, Debug, PartialEq)]
pub struct MmParams {
    pub index_min: usize,
    pub index_max: usize,
    pub min_distance: usize,
    pub window: usize,
    pub tp_th: f64,
    pub tp_limit: f64,
    pub signal_hold: f64,
    pub signal_short: f64,
    pub signal_long: f64,
}

impl Default for MmParams {
    fn default() -> Self {
        Self {
            index_min: 0,
            index_max: 0,
            min_distance: 10,
            tp_th: 0.03,
            tp_limit: 0.07,
            signal_hold: 0.,
            signal_short: -1.,
            signal_long: 1.,
            window: 60,
        }
    }
}

impl MmParams {
    pub fn new(
        index_min: usize,
        index_max: usize,
        min_distance: usize,
        window: usize,
        tp_th: f64,
        tp_limit: f64,
        signal_hold: f64,
        signal_short: f64,
        signal_long: f64,
    ) -> Self {
        Self {
            index_min,
            index_max,
            min_distance,
            tp_th,
            tp_limit,
            signal_hold,
            signal_short,
            signal_long,
            window,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Default)]
pub struct MmBf {
    pub src_l: Vec<Vec<f64>>,
}

#[derive(Clone, Debug, PartialEq, Default)]
pub struct MM {
    pub params: MmParams,
    bf: RefCell<MmBf>,
    bf_state: RefCell<MmBf>,
}

impl MM {
    pub fn new(
        index_min: usize,
        index_max: usize,
        min_distance: usize,
        window: usize,
        tp_th: f64,
        tp_limit: f64,
        signal_hold: f64,
        signal_short: f64,
        signal_long: f64,
    ) -> Self {
        Self {
            params: MmParams::new(
                index_min,
                index_max,
                min_distance,
                window,
                tp_th,
                tp_limit,
                signal_hold,
                signal_short,
                signal_long,
            ),
            ..Default::default()
        }
    }
}

impl SignalTrain for MM {
    fn w(&self) -> usize {
        self.params.window + 1
    }
    fn init_bf(&self, src: &[Vec<f64>]) {
        self.bf.borrow_mut().src_l = src[src.len() - self.params.window..].to_vec();
    }
    fn execute_bf(&self) {
        *self.bf.borrow_mut() = self.bf_state.borrow().clone();
    }
    fn signal_with_bf(&self, src: &[f64]) -> f64 {
        self.bf_state.borrow_mut().src_l = self.bf.borrow_mut().src_l[1..].to_vec();
        self.bf_state.borrow_mut().src_l.push(src.to_vec());
        let bind = self.bf_state.borrow();
        let v = bind.src_l.iter().cloned().enumerate();
        let min_ = (
            v.clone()
                .min_by(|v1, v2| {
                    v1.1[self.params.index_min]
                        .partial_cmp(&v2.1[self.params.index_min])
                        .unwrap_or(Equal)
                })
                .unwrap_or_default(),
            self.params.signal_long,
        );
        let max_ = (
            v.clone()
                .max_by(|v1, v2| {
                    v1.1[self.params.index_max]
                        .partial_cmp(&v2.1[self.params.index_max])
                        .unwrap_or(Equal)
                })
                .unwrap_or_default(),
            self.params.signal_short,
        );
        let percent = (max_.0.1[self.params.index_max] - min_.0.1[self.params.index_min])
            / max_.0.1[self.params.index_max];
        if percent >= self.params.tp_th && percent <= self.params.tp_limit {
            if self.params.min_distance <= (min_.0.0.max(max_.0.0) - max_.0.0.min(min_.0.0)) {
                let res = min_by_key(max_, min_, |v1| v1.0.0);
                if res.0.0 == 0 {
                    return res.1;
                }
                return self.params.signal_hold;
            }
            return self.params.signal_hold;
        }
        self.params.signal_hold
    }
}

impl SignalTrainExt for MM {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude_tests::prelude::*;

    const RES: f64 = 0.0;
    static SIGNAL: LazyLock<fn() -> MM> =
        LazyLock::new(|| || MM::new(1, 1, 2, 3, 0.0001, 0.01, 0.0, 1.0, 2.0));

    #[test]
    fn mm_with_bf_res_1() {
        test_bf_res_1(&SIGNAL(), &SRC, RES);
    }

    #[test]
    fn mm_signal_res_1() {
        test_f_res_1(&SIGNAL(), &SRC, RES);
    }

    #[test]
    fn mm_coll_res_1() {
        test_coll_res_1(&SIGNAL(), &SRC, RES, 30);
    }

    #[test]
    fn mm_coll_res_2() {
        test_coll_res_2(&SIGNAL(), &SRC, 30);
    }

    #[test]
    fn mm_coll_res_3() {
        test_coll_res_3(
            &SIGNAL(),
            &SRC,
            vec![
                0.0,
                2.0,
                2.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                1.0,
                1.0,
                1.0,
                0.0,
                2.0,
                0.0,
                0.0,
                2.0,
                0.0,
                0.0,
                2.0,
                2.0,
                2.0,
                0.0,
                1.0,
                0.0,
                2.0,
                2.0,
                0.0,
                1.0,
                0.0,
                0.0,
                1.0,
                1.0,
                1.0,
                0.0,
                2.0,
                0.0,
                1.0,
                1.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                f64::NAN,
                f64::NAN,
                f64::NAN,
            ],
        );
    }
}
