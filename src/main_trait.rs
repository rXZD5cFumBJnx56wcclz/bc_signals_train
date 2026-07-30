use std::any::Any;

use bc_utils_lg::traits::w::W;
use dyn_clone::DynClone;

fn signal_coll<C, T>(signal_struct: &T, src: &[Vec<f64>]) -> C
where
    C: FromIterator<f64>,
    T: SignalTrain,
    T: ?Sized,
{
    src.iter()
        .map(|v| {
            let bind = signal_struct.signal(v);
            signal_struct.execute_bf();
            bind
        })
        .collect()
}

pub trait SignalTrain: Any + W + DynClone {
    fn init_bf(&self, src: &[Vec<f64>]);
    fn execute_bf(&self);
    fn signal(&self, src: &[f64]) -> f64;
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
