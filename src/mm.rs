use std::cmp::{Ordering::Equal, min_by_key};

use crate::prelude::*;

#[derive(Clone, Debug, PartialEq)]
pub struct MM {
    pub index_min: usize,
    pub index_max: usize,
    pub min_distance: usize,
    pub tp_th: f64,
    pub tp_limit: f64,
    pub signal_hold: f64,
    pub signal_short: f64,
    pub signal_long: f64,
    pub window: usize,
    pub mult_window_accuracy: usize,
    pub add_window_accuracy: usize,
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
            window: window,
            mult_window_accuracy: 1,
            add_window_accuracy: 1,
            index_min,
            index_max,
            min_distance,
            tp_th,
            tp_limit,
            signal_hold,
            signal_short,
            signal_long,
        }
    }
    pub fn set_window(
        &mut self,
        window: usize,
    ) {
        self.window = window;
    }
    pub fn set_mult_window_accuracy(
        &mut self,
        mult_window_accuracy: usize,
    ) {
        self.mult_window_accuracy = mult_window_accuracy;
    }
    pub fn set_add_window_accuracy(
        &mut self,
        add_window_accuracy: usize,
    ) {
        self.add_window_accuracy = add_window_accuracy;
    }
    pub fn set_signal_hold(
        &mut self,
        signal_hold: f64,
    ) {
        self.signal_hold = signal_hold;
    }
    pub fn set_signal_short(
        &mut self,
        signal_short: f64,
    ) {
        self.signal_short = signal_short;
    }
    pub fn set_signal_long(
        &mut self,
        signal_long: f64,
    ) {
        self.signal_long = signal_long;
    }
    pub fn set_index_min(
        &mut self,
        index_min: usize,
    ) {
        self.index_min = index_min;
    }
    pub fn set_index_max(
        &mut self,
        index_max: usize,
    ) {
        self.index_max = index_max;
    }
    pub fn set_min_distance(
        &mut self,
        min_distance: usize,
    ) {
        self.min_distance = min_distance;
    }
    pub fn set_tp_th(
        &mut self,
        tp_th: f64,
    ) {
        self.tp_th = tp_th;
    }
    pub fn set_tp_limit(
        &mut self,
        tp_limit: f64,
    ) {
        self.tp_limit = tp_limit;
    }
}

impl Default for MM {
    fn default() -> Self {
        MM::new(0, 0, 10, 60, 0.03, 0.05, 0.0, -1.0, 1.0)
    }
}

impl SignalTrain for MM {
    fn w(&self) -> usize {
        self.window * self.mult_window_accuracy + self.add_window_accuracy
    }
    fn bf<'a>(
        &self,
        src: &[Vec<f64>],
        _: &[Vec<f64>],
    ) -> BF_SIGNALS<'a> {
        <BF_SIGNALS as BfSignalsExt>::new([(
            "src_l_vec",
            src[src.len() - (self.w() - 1)..].to_vec(),
        )])
    }
    fn signal_with_bf<'a>(
        &self,
        src: &[f64],
        _: &[f64],
        bf: &BF_SIGNALS<'a>,
        index_: usize,
    ) -> f64 {
        bf.roll_and_replace(-1, index_, "src_l_vec", src.to_vec());
        let bind = bf.borrow();
        let v = bind[index_]["src_l_vec"].iter().cloned().enumerate();
        let min_ = (
            v.clone()
                .min_by(|v1, v2| {
                    v1.1[self.index_min]
                        .partial_cmp(&v2.1[self.index_min])
                        .unwrap_or(Equal)
                })
                .unwrap_or_default(),
            self.signal_long,
        );
        let max_ = (
            v.clone()
                .max_by(|v1, v2| {
                    v1.1[self.index_max]
                        .partial_cmp(&v2.1[self.index_max])
                        .unwrap_or(Equal)
                })
                .unwrap_or_default(),
            self.signal_short,
        );
        let percent =
            (max_.0.1[self.index_max] - min_.0.1[self.index_min]) / max_.0.1[self.index_max];
        if percent >= self.tp_th && percent <= self.tp_limit {
            if self.min_distance <= (min_.0.0.max(max_.0.0) - max_.0.0.min(min_.0.0)) {
                let res = min_by_key(max_, min_, |v1| v1.0.0);
                if res.0.0 == 0 {
                    return res.1;
                }
                return self.signal_hold;
            }
            return self.signal_hold;
        }
        self.signal_hold
    }
}

impl SignalTrainExt for MM {}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::test_funcs::test_funcs::*;

    const RES: f64 = 0.0;
    static SIGNALS: LazyLock<Vec<Vec<f64>>> = LazyLock::new(|| Default::default());
    static SIGNAL: LazyLock<MM> =
        LazyLock::new(|| MM::new(1, 1, 2, 3, 0.0001, 0.01, 0.0, 1.0, 2.0));

    #[test]
    fn mm_with_bf_res_1() {
        test_bf_res_1(&*SIGNAL, &SRC, &SIGNALS, RES);
    }

    #[test]
    fn mm_signal_res_1() {
        test_f_res_1(&*SIGNAL, &SRC, &SIGNALS, RES);
    }

    #[test]
    fn mm_coll_res_1() {
        test_coll_res_1(&*SIGNAL, &SRC, &SIGNALS, RES, 30);
    }

    #[test]
    fn mm_coll_res_2() {
        test_coll_res_2(&*SIGNAL, &SRC, &SIGNALS, 30);
    }

    #[test]
    fn mm_coll_res_3() {
        test_coll_res_3(
            &*SIGNAL,
            &SRC,
            &SIGNALS,
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
