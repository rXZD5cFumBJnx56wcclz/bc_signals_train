use std::any::Any;

use dyn_clone::DynClone;

fn signal_coll<C, T>(signal_struct: &T, src: &[Vec<f64>]) -> C
where
    C: FromIterator<f64>,
    T: SignalTrain,
    T: ?Sized,
{
    let w = signal_struct.w().checked_sub(1).unwrap_or_default();
    signal_struct.init_bf(&src.get(..w).unwrap_or_default());
    src.iter()
        .skip(w)
        .map(|v| {
            let bind = signal_struct.signal_with_bf(v);
            signal_struct.execute_bf();
            bind
        })
        .chain(std::iter::repeat(f64::NAN).take(w))
        .collect()
}

pub trait SignalTrain: Any + DynClone {
    fn w(&self) -> usize;
    fn init_bf(&self, src: &[Vec<f64>]);
    fn execute_bf(&self);
    fn signal_with_bf(&self, src: &[f64]) -> f64;
    fn signal(&self, src: &[Vec<f64>]) -> f64 {
        self.init_bf(
            &src[src.len().checked_sub(self.w()).unwrap_or_default()
                ..src.len().checked_sub(1).unwrap_or_default()],
        );
        self.signal_with_bf(src.last().unwrap_or(&vec![]))
    }
    fn signals_vec(&self, src: &[Vec<f64>]) -> Vec<f64> {
        signal_coll(self, src)
    }
}

dyn_clone::clone_trait_object!(SignalTrain);

pub trait SignalTrainExt: SignalTrain {
    fn signal_coll<C>(&self, src: &[Vec<f64>]) -> C
    where
        C: FromIterator<f64>,
    {
        signal_coll(self, src)
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
