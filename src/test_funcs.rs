#[cfg(test)]
pub mod test_funcs {
    use crate::prelude::*;
    use pretty_assertions::assert_eq as assert_eq_pr;

    pub fn test_bf_res_1<T>(settings_signal: &T, in_: &[Vec<f64>], eq: f64)
    where
        T: SignalTrain,
        T: SignalTrainExt,
    {
        settings_signal.init_bf(
            in_.get(..in_.len().checked_sub(1).unwrap_or_default())
                .unwrap_or_default(),
        );
        assert_eq_pr!(settings_signal.signal(in_.last().unwrap(),), eq,);
    }

    pub fn test_coll_res_1<T>(settings_signal: &T, in_: &[Vec<f64>], interval_len: usize)
    where
        T: SignalTrain,
        T: Clone,
        T: SignalTrainExt,
    {
        let sign_vec = settings_signal.clone();
        sign_vec.init_bf(&in_[..in_.len() - interval_len]);
        let sign_value = settings_signal.clone();
        sign_value.init_bf(&in_[..in_.len() - 1]);
        assert_eq_pr!(
            sign_vec
                .signals_vec(&in_[in_.len() - interval_len..])
                .last()
                .copied()
                .unwrap(),
            sign_value.signal(&in_.last().cloned().unwrap()),
        );
    }
}
