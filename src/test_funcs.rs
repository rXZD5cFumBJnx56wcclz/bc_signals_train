#[cfg(test)]
pub mod test_funcs {
    use crate::prelude::*;
    use bc_utils::nums::nz_coll;
    use pretty_assertions::assert_eq as assert_eq_pr;

    pub fn test_bf_res_1<T>(
        settings_signal: &T,
        in_: &[Vec<f64>],
        signals: &[Vec<f64>],
        eq: f64,
    ) where
        T: SignalTrain,
        T: SignalTrainExt,
    {
        let len_sub_in = in_.len().checked_sub(1).unwrap_or_default();
        let len_sub_signals = signals.len().checked_sub(1).unwrap_or_default();
        let bf = settings_signal.bf(
            in_.get(..len_sub_in).unwrap_or_default(),
            signals.get(..len_sub_signals).unwrap_or_default(),
        );
        assert_eq_pr!(
            settings_signal.signal_with_bf(
                in_.last().unwrap_or(&vec![0.0]),
                &signals.get(len_sub_signals).unwrap_or(&vec![]),
                &bf,
                0
            ),
            eq,
        );
    }

    pub fn test_f_res_1<T>(
        settings_signal: &T,
        in_: &[Vec<f64>],
        signals: &[Vec<f64>],
        eq: f64,
    ) where
        T: SignalTrain,
        T: SignalTrainExt,
    {
        assert_eq_pr!(settings_signal.signal(in_, signals,), eq,);
    }

    pub fn test_coll_res_1<T>(
        settings_signal: &T,
        in_: &[Vec<f64>],
        signals: &[Vec<f64>],
        eq: f64,
        len_elements: usize,
    ) where
        T: SignalTrain,
        T: SignalTrainExt,
    {
        assert_eq_pr!(
            dbg!(
                settings_signal.signal_coll::<Vec<_>>(
                    &in_.get(in_.len().checked_sub(len_elements).unwrap_or_default()..)
                        .unwrap_or_default(),
                    signals,
                )
            )[len_elements - 1 - settings_signal.w()],
            eq,
        );
    }

    pub fn test_coll_res_2<T>(
        settings_signal: &T,
        in_: &[Vec<f64>],
        signals: &[Vec<f64>],
        len_elements: usize,
    ) where
        T: SignalTrainExt,
    {
        let in_ = &in_
            .get(in_.len().checked_sub(len_elements).unwrap_or_default()..)
            .unwrap_or_default();
        assert_eq_pr!(
            settings_signal.signal_coll::<Vec<_>>(in_, signals,)
                [len_elements - 1 - settings_signal.w()],
            settings_signal.signal(in_, signals,),
        );
    }

    pub fn test_coll_res_3<T>(
        settings_signal: &T,
        in_: &[Vec<f64>],
        signals: &[Vec<f64>],
        eq: Vec<f64>,
    ) where
        T: SignalTrain,
        T: SignalTrainExt,
    {
        assert_eq_pr!(
            nz_coll::<Vec<f64>, f64, _>(&settings_signal.signals_vec(in_, signals), 0.0),
            nz_coll::<Vec<f64>, f64, _>(&eq, 0.0),
        );
    }
}
