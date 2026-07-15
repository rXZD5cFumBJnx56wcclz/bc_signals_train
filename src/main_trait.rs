use std::any::Any;

use bc_signals::bf_signals::BF_SIGNALS;

fn signal_coll<C, T>(
    signal_struct: &T,
    src: &[Vec<f64>],
    signals: &[Vec<f64>],
) -> C
where
    C: FromIterator<f64>,
    T: SignalTrain,
    T: ?Sized,
{
    let w = signal_struct.w().checked_sub(1).unwrap_or_default();
    let bf = signal_struct.bf(
        &src.get(..w).unwrap_or_default(),
        signals.get(..w).unwrap_or_default(),
    );
    match (src.is_empty(), signals.is_empty()) {
        (false, false) | (true, true) => src
            .iter()
            .zip(signals)
            .skip(w)
            .map(|(sr, s)| signal_struct.signal_with_bf(sr, s, &bf, 0))
            .chain(std::iter::repeat(f64::NAN).take(w))
            .collect(),
        (true, false) => signals
            .iter()
            .skip(w)
            .map(|s| signal_struct.signal_with_bf(Default::default(), s, &bf, 0))
            .chain(std::iter::repeat(f64::NAN).take(w))
            .collect(),
        (false, true) => src
            .iter()
            .skip(w)
            .map(|sr| signal_struct.signal_with_bf(sr, Default::default(), &bf, 0))
            .chain(std::iter::repeat(f64::NAN).take(w))
            .collect(),
    }
}

pub trait SignalTrain: Any {
    fn w(&self) -> usize;
    fn bf<'a>(
        &self,
        src: &[Vec<f64>],
        signals: &[Vec<f64>],
    ) -> BF_SIGNALS<'a>;
    fn signal_with_bf<'a>(
        &self,
        src: &[f64],
        signals: &[f64],
        bf: &BF_SIGNALS<'a>,
        index_: usize,
    ) -> f64;
    fn signal(
        &self,
        src: &[Vec<f64>],
        signals: &[Vec<f64>],
    ) -> f64 {
        let bf = self.bf(
            &src[src.len().checked_sub(self.w()).unwrap_or_default()
                ..src.len().checked_sub(1).unwrap_or_default()],
            &signals[signals.len().checked_sub(self.w()).unwrap_or_default()
                ..signals.len().checked_sub(1).unwrap_or_default()],
        );
        self.signal_with_bf(
            src.last().unwrap_or(&vec![]),
            signals.last().unwrap_or(&vec![]),
            &bf,
            0,
        )
    }
    fn signals_vec(
        &self,
        src: &[Vec<f64>],
        signals: &[Vec<f64>],
    ) -> Vec<f64> {
        signal_coll(self, src, signals)
    }
}

pub trait SignalTrainExt: SignalTrain {
    fn signal_coll<C>(
        &self,
        src: &[Vec<f64>],
        signals: &[Vec<f64>],
    ) -> C
    where
        C: FromIterator<f64>,
    {
        signal_coll(self, src, signals)
    }
}

type SignalTrainType = Vec<f64>;

pub trait SignalTrainTo {
    fn to_i32(self) -> Vec<i32>;
    fn to_i64(self) -> Vec<i64>;
}

impl SignalTrainTo for SignalTrainType {
    fn to_i32(self) -> Vec<i32> {
        self.into_iter().map(|v| v as i32).collect()
    }
    fn to_i64(self) -> Vec<i64> {
        self.into_iter().map(|v| v as i64).collect()
    }
}
