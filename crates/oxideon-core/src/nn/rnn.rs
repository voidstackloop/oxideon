//! Recurrent Neural Network building blocks: [`RNNCell`], [`LSTMCell`], [`GRUCell`],
//! and sequence-level [`RNN`], [`LSTM`], and [`GRU`].

use super::{Linear, Module};
use crate::error::{Error, Result};
use crate::tensor::Tensor;

// ---------------------------------------------------------------------------
// RNNCell: h' = tanh(x @ W_ih^T + b_ih + h @ W_hh^T + b_hh)
// ---------------------------------------------------------------------------
pub struct RNNCell {
    pub ih: Linear,
    pub hh: Linear,
    pub input_size: usize,
    pub hidden_size: usize,
}

impl RNNCell {
    pub fn new(input_size: usize, hidden_size: usize) -> Result<Self> {
        Ok(Self {
            ih: Linear::new(input_size, hidden_size)?,
            hh: Linear::new(hidden_size, hidden_size)?,
            input_size,
            hidden_size,
        })
    }

    pub fn step(&self, x: &Tensor, h: Option<&Tensor>) -> Result<Tensor> {
        if x.ndim() != 2 {
            return Err(Error::Msg(format!(
                "RNNCell::step expects 2D input [B, input_size], got {:?}",
                x.shape()
            )));
        }
        let b = x.shape()[0];
        let h_prev = match h {
            Some(h) => h.clone(),
            None => Tensor::zeros(&[b, self.hidden_size], x.dtype(), &x.device())?,
        };
        self.ih.forward(x)?.add(&self.hh.forward(&h_prev)?)?.tanh()
    }

    pub fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut p = Vec::new();
        for (k, v) in self.ih.named_parameters() {
            p.push((format!("ih.{k}"), v));
        }
        for (k, v) in self.hh.named_parameters() {
            p.push((format!("hh.{k}"), v));
        }
        p
    }
}

// ---------------------------------------------------------------------------
// LSTMCell: fused 4-gate projection [i, f, g, o] of size 4 * hidden_size
// ---------------------------------------------------------------------------
pub struct LSTMCell {
    pub ih: Linear,
    pub hh: Linear,
    pub input_size: usize,
    pub hidden_size: usize,
}

impl LSTMCell {
    pub fn new(input_size: usize, hidden_size: usize) -> Result<Self> {
        let gates = 4usize.checked_mul(hidden_size).ok_or_else(|| {
            Error::Msg(format!(
                "LSTMCell: gate dim 4 * hidden_size ({hidden_size}) overflows"
            ))
        })?;
        Ok(Self {
            ih: Linear::new(input_size, gates)?,
            hh: Linear::new(hidden_size, gates)?,
            input_size,
            hidden_size,
        })
    }

    pub fn step(&self, x: &Tensor, state: Option<(&Tensor, &Tensor)>) -> Result<(Tensor, Tensor)> {
        if x.ndim() != 2 {
            return Err(Error::Msg(format!(
                "LSTMCell::step expects 2D input [B, input_size], got {:?}",
                x.shape()
            )));
        }
        let b = x.shape()[0];
        let hs = self.hidden_size;
        let (h_prev, c_prev) = match state {
            Some((h, c)) => (h.clone(), c.clone()),
            None => (
                Tensor::zeros(&[b, hs], x.dtype(), &x.device())?,
                Tensor::zeros(&[b, hs], x.dtype(), &x.device())?,
            ),
        };
        let gates = self.ih.forward(x)?.add(&self.hh.forward(&h_prev)?)?;
        let i_gate = gates.narrow(1, 0, hs)?.sigmoid()?;
        let f_gate = gates.narrow(1, hs, hs)?.sigmoid()?;
        let g_gate = gates.narrow(1, 2 * hs, hs)?.tanh()?;
        let o_gate = gates.narrow(1, 3 * hs, hs)?.sigmoid()?;

        let c_next = f_gate.mul(&c_prev)?.add(&i_gate.mul(&g_gate)?)?;
        let h_next = o_gate.mul(&c_next.tanh()?)?;
        Ok((h_next, c_next))
    }

    pub fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut p = Vec::new();
        for (k, v) in self.ih.named_parameters() {
            p.push((format!("ih.{k}"), v));
        }
        for (k, v) in self.hh.named_parameters() {
            p.push((format!("hh.{k}"), v));
        }
        p
    }
}

// ---------------------------------------------------------------------------
// GRUCell: fused 3-gate projection [r, z, n] of size 3 * hidden_size
// ---------------------------------------------------------------------------
pub struct GRUCell {
    pub ih: Linear,
    pub hh: Linear,
    pub input_size: usize,
    pub hidden_size: usize,
}

impl GRUCell {
    pub fn new(input_size: usize, hidden_size: usize) -> Result<Self> {
        let gates = 3usize.checked_mul(hidden_size).ok_or_else(|| {
            Error::Msg(format!(
                "GRUCell: gate dim 3 * hidden_size ({hidden_size}) overflows"
            ))
        })?;
        Ok(Self {
            ih: Linear::new(input_size, gates)?,
            hh: Linear::new(hidden_size, gates)?,
            input_size,
            hidden_size,
        })
    }

    pub fn step(&self, x: &Tensor, h: Option<&Tensor>) -> Result<Tensor> {
        if x.ndim() != 2 {
            return Err(Error::Msg(format!(
                "GRUCell::step expects 2D input [B, input_size], got {:?}",
                x.shape()
            )));
        }
        let b = x.shape()[0];
        let hs = self.hidden_size;
        let h_prev = match h {
            Some(h) => h.clone(),
            None => Tensor::zeros(&[b, hs], x.dtype(), &x.device())?,
        };
        let gi = self.ih.forward(x)?;
        let gh = self.hh.forward(&h_prev)?;

        let r_gate = gi.narrow(1, 0, hs)?.add(&gh.narrow(1, 0, hs)?)?.sigmoid()?;
        let z_gate = gi
            .narrow(1, hs, hs)?
            .add(&gh.narrow(1, hs, hs)?)?
            .sigmoid()?;
        let n_gate = gi
            .narrow(1, 2 * hs, hs)?
            .add(&r_gate.mul(&gh.narrow(1, 2 * hs, hs)?)?)?
            .tanh()?;

        // h' = (1 - z) * n + z * h_prev
        let one_minus_z = z_gate.neg()?.add_scalar(1.0)?;
        one_minus_z.mul(&n_gate)?.add(&z_gate.mul(&h_prev)?)
    }

    pub fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut p = Vec::new();
        for (k, v) in self.ih.named_parameters() {
            p.push((format!("ih.{k}"), v));
        }
        for (k, v) in self.hh.named_parameters() {
            p.push((format!("hh.{k}"), v));
        }
        p
    }
}

// ---------------------------------------------------------------------------
// Sequence-level RNN, LSTM, and GRU (`input: [batch, seq_len, input_size]`)
// ---------------------------------------------------------------------------

pub struct RNN {
    pub cells: Vec<RNNCell>,
    pub input_size: usize,
    pub hidden_size: usize,
}

impl RNN {
    pub fn new(input_size: usize, hidden_size: usize, num_layers: usize) -> Result<Self> {
        if num_layers < 1 {
            return Err(crate::error::Error::Msg(
                "num_layers must be >= 1".to_string(),
            ));
        }
        let mut cells = crate::layout::try_with_capacity(num_layers)?;
        crate::limits::check_repeats("num_layers", num_layers)?;
        for l in 0..num_layers {
            let in_dim = if l == 0 { input_size } else { hidden_size };
            cells.push(RNNCell::new(in_dim, hidden_size)?);
        }
        Ok(Self {
            cells,
            input_size,
            hidden_size,
        })
    }

    /// Runs the multi-layer RNN over `input: [batch, seq_len, input_size]`, returning
    /// `(output_seq: [batch, seq_len, hidden_size], h_n: [num_layers, batch, hidden_size])`.
    /// Like torch, a provided `h0` must be exactly `[num_layers, batch, hidden_size]`.
    pub fn forward_seq(&self, input: &Tensor, h0: Option<&Tensor>) -> Result<(Tensor, Tensor)> {
        if input.ndim() != 3 {
            return Err(Error::Msg(
                "RNN expects [batch, seq_len, input_size]".into(),
            ));
        }
        if let Some(h_init) = h0 {
            let want = vec![self.cells.len(), input.shape()[0], self.hidden_size];
            if h_init.shape() != want.as_slice() {
                return Err(Error::ShapeMismatch {
                    op: "RNN::forward_seq h0",
                    expected: want,
                    got: h_init.shape().to_vec(),
                });
            }
        }
        let seq_len = input.shape()[1];
        let mut cur_seq = input.clone();
        let mut final_hs = Vec::with_capacity(self.cells.len());

        for (l, cell) in self.cells.iter().enumerate() {
            let mut h = match h0 {
                Some(h_init) => Some(h_init.select(0, l as isize)?),
                None => None,
            };
            let mut step_outs = Vec::with_capacity(seq_len);
            for t in 0..seq_len {
                let xt = cur_seq.select(1, t as isize)?;
                let h_next = cell.step(&xt, h.as_ref())?;
                step_outs.push(h_next.clone());
                h = Some(h_next);
            }
            cur_seq = Tensor::stack(&step_outs, 1)?;
            final_hs.push(h.unwrap());
        }
        let h_n = Tensor::stack(&final_hs, 0)?;
        Ok((cur_seq, h_n))
    }
}

impl Module for RNN {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        Ok(self.forward_seq(input, None)?.0)
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut out = Vec::new();
        for (l, cell) in self.cells.iter().enumerate() {
            for (k, v) in cell.named_parameters() {
                out.push((format!("layer{l}.{k}"), v));
            }
        }
        out
    }
}

pub struct LSTM {
    pub cells: Vec<LSTMCell>,
    pub input_size: usize,
    pub hidden_size: usize,
}

impl LSTM {
    pub fn new(input_size: usize, hidden_size: usize, num_layers: usize) -> Result<Self> {
        if num_layers < 1 {
            return Err(crate::error::Error::Msg(
                "num_layers must be >= 1".to_string(),
            ));
        }
        let mut cells = crate::layout::try_with_capacity(num_layers)?;
        crate::limits::check_repeats("num_layers", num_layers)?;
        for l in 0..num_layers {
            let in_dim = if l == 0 { input_size } else { hidden_size };
            cells.push(LSTMCell::new(in_dim, hidden_size)?);
        }
        Ok(Self {
            cells,
            input_size,
            hidden_size,
        })
    }

    /// Runs the multi-layer LSTM over `input: [batch, seq_len, input_size]`, returning
    /// `(output_seq: [batch, seq_len, hidden_size], (h_n, c_n))` where `h_n` and `c_n`
    /// have shape `[num_layers, batch, hidden_size]`. Like torch, provided `h0`/`c0`
    /// must each be exactly `[num_layers, batch, hidden_size]`.
    pub fn forward_seq(
        &self,
        input: &Tensor,
        state0: Option<(&Tensor, &Tensor)>,
    ) -> Result<(Tensor, (Tensor, Tensor))> {
        if input.ndim() != 3 {
            return Err(Error::Msg(
                "LSTM expects [batch, seq_len, input_size]".into(),
            ));
        }
        if let Some((h_init, c_init)) = state0 {
            let want = vec![self.cells.len(), input.shape()[0], self.hidden_size];
            for (name, state) in [("h0", h_init), ("c0", c_init)] {
                if state.shape() != want.as_slice() {
                    return Err(Error::ShapeMismatch {
                        op: if name == "h0" {
                            "LSTM::forward_seq h0"
                        } else {
                            "LSTM::forward_seq c0"
                        },
                        expected: want.clone(),
                        got: state.shape().to_vec(),
                    });
                }
            }
        }
        let seq_len = input.shape()[1];
        let mut cur_seq = input.clone();
        let mut final_hs = Vec::with_capacity(self.cells.len());
        let mut final_cs = Vec::with_capacity(self.cells.len());

        for (l, cell) in self.cells.iter().enumerate() {
            let mut state = match state0 {
                Some((h0, c0)) => Some((h0.select(0, l as isize)?, c0.select(0, l as isize)?)),
                None => None,
            };
            let mut step_outs = Vec::with_capacity(seq_len);
            for t in 0..seq_len {
                let xt = cur_seq.select(1, t as isize)?;
                let (h_next, c_next) = cell.step(&xt, state.as_ref().map(|(h, c)| (h, c)))?;
                step_outs.push(h_next.clone());
                state = Some((h_next, c_next));
            }
            cur_seq = Tensor::stack(&step_outs, 1)?;
            let (hl, cl) = state.unwrap();
            final_hs.push(hl);
            final_cs.push(cl);
        }
        Ok((
            cur_seq,
            (Tensor::stack(&final_hs, 0)?, Tensor::stack(&final_cs, 0)?),
        ))
    }
}

impl Module for LSTM {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        Ok(self.forward_seq(input, None)?.0)
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut out = Vec::new();
        for (l, cell) in self.cells.iter().enumerate() {
            for (k, v) in cell.named_parameters() {
                out.push((format!("layer{l}.{k}"), v));
            }
        }
        out
    }
}

pub struct GRU {
    pub cells: Vec<GRUCell>,
    pub input_size: usize,
    pub hidden_size: usize,
}

impl GRU {
    pub fn new(input_size: usize, hidden_size: usize, num_layers: usize) -> Result<Self> {
        if num_layers < 1 {
            return Err(crate::error::Error::Msg(
                "num_layers must be >= 1".to_string(),
            ));
        }
        let mut cells = crate::layout::try_with_capacity(num_layers)?;
        crate::limits::check_repeats("num_layers", num_layers)?;
        for l in 0..num_layers {
            let in_dim = if l == 0 { input_size } else { hidden_size };
            cells.push(GRUCell::new(in_dim, hidden_size)?);
        }
        Ok(Self {
            cells,
            input_size,
            hidden_size,
        })
    }

    /// Like torch, a provided `h0` must be exactly `[num_layers, batch, hidden_size]`.
    pub fn forward_seq(&self, input: &Tensor, h0: Option<&Tensor>) -> Result<(Tensor, Tensor)> {
        if input.ndim() != 3 {
            return Err(Error::Msg(
                "GRU expects [batch, seq_len, input_size]".into(),
            ));
        }
        if let Some(h_init) = h0 {
            let want = vec![self.cells.len(), input.shape()[0], self.hidden_size];
            if h_init.shape() != want.as_slice() {
                return Err(Error::ShapeMismatch {
                    op: "GRU::forward_seq h0",
                    expected: want,
                    got: h_init.shape().to_vec(),
                });
            }
        }
        let seq_len = input.shape()[1];
        let mut cur_seq = input.clone();
        let mut final_hs = Vec::with_capacity(self.cells.len());

        for (l, cell) in self.cells.iter().enumerate() {
            let mut h = match h0 {
                Some(h_init) => Some(h_init.select(0, l as isize)?),
                None => None,
            };
            let mut step_outs = Vec::with_capacity(seq_len);
            for t in 0..seq_len {
                let xt = cur_seq.select(1, t as isize)?;
                let h_next = cell.step(&xt, h.as_ref())?;
                step_outs.push(h_next.clone());
                h = Some(h_next);
            }
            cur_seq = Tensor::stack(&step_outs, 1)?;
            final_hs.push(h.unwrap());
        }
        let h_n = Tensor::stack(&final_hs, 0)?;
        Ok((cur_seq, h_n))
    }
}

impl Module for GRU {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        Ok(self.forward_seq(input, None)?.0)
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut out = Vec::new();
        for (l, cell) in self.cells.iter().enumerate() {
            for (k, v) in cell.named_parameters() {
                out.push((format!("layer{l}.{k}"), v));
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(data: &[f64], shape: &[usize]) -> Tensor {
        Tensor::from_slice(data, shape).unwrap()
    }

    /// Row-distinct gate weights: row r = [0.1(r+1), -0.05(r+1)], so any gate
    /// permutation changes every output (matches the torch reference probe).
    fn wmat(rows: usize) -> Tensor {
        let mut v = Vec::with_capacity(rows * 2);
        for r in 0..rows {
            v.push(0.1 * (r + 1) as f64);
            v.push(-0.05 * (r + 1) as f64);
        }
        t(&v, &[rows, 2])
    }

    fn bias_vec(n: usize, scale: f64) -> Tensor {
        t(
            &(0..n)
                .map(|r| 0.01 * (r + 1) as f64 * scale)
                .collect::<Vec<_>>(),
            &[n],
        )
    }

    /// Builds an F64 `Linear` with exact weights (assign-based injection would
    /// cast down to the default F32 params).
    fn lin(w: &Tensor, b: &Tensor) -> Linear {
        Linear::from_tensors(w.clone(), Some(b.clone())).unwrap()
    }

    fn x0() -> Tensor {
        t(&[0.5, -1.5, 1.0, 0.25], &[2, 2])
    }

    fn h0() -> Tensor {
        t(&[0.2, 0.3, -0.4, 0.1], &[2, 2])
    }

    fn c0() -> Tensor {
        t(&[0.1, -0.2, 0.3, 0.4], &[2, 2])
    }

    fn assert_close(got: &Tensor, want: &[f64]) {
        let g = got.to_vec::<f64>().unwrap();
        assert_eq!(g.len(), want.len());
        for (a, b) in g.iter().zip(want.iter()) {
            assert!((a - b).abs() < 1e-9, "{a} vs {b}");
        }
    }

    #[test]
    fn rnn_cell_matches_torch() {
        let mut cell = RNNCell::new(2, 2).unwrap();
        cell.ih = lin(&wmat(2), &bias_vec(2, 1.0));
        cell.hh = lin(&wmat(2).mul_scalar(2.0).unwrap(), &bias_vec(2, 3.0));
        let out = cell.step(&x0(), Some(&h0())).unwrap();
        assert_eq!(out.shape(), &[2, 2]);
        assert_close(
            &out,
            &[
                0.173235157835,
                0.336375544336,
                0.037482431757,
                0.074859690687,
            ],
        );
    }

    #[test]
    fn lstm_cell_matches_torch() {
        let mut cell = LSTMCell::new(2, 2).unwrap();
        cell.ih = lin(&wmat(8), &bias_vec(8, 1.0));
        cell.hh = lin(&wmat(8).mul_scalar(2.0).unwrap(), &bias_vec(8, 3.0));
        let (h, c) = cell.step(&x0(), Some((&h0(), &c0()))).unwrap();
        assert_close(
            &h,
            &[
                0.323240269082,
                0.251890811742,
                0.139944445732,
                0.182847058922,
            ],
        );
        assert_close(
            &c,
            &[
                0.445501937292,
                0.324983798249,
                0.252832507441,
                0.329758199623,
            ],
        );
    }

    #[test]
    fn gru_cell_matches_torch() {
        let mut cell = GRUCell::new(2, 2).unwrap();
        cell.ih = lin(&wmat(6), &bias_vec(6, 1.0));
        cell.hh = lin(&wmat(6).mul_scalar(2.0).unwrap(), &bias_vec(6, 3.0));
        let out = cell.step(&x0(), Some(&h0())).unwrap();
        assert_close(
            &out,
            &[
                0.369057083797,
                0.446043729947,
                -0.058941932201,
                0.228804190206,
            ],
        );
    }

    /// 2-layer batch-first sequence parity vs `torch.nn.{RNN,LSTM,GRU}`
    /// (batch_first=True): outputs plus final states, with per-layer weights.
    #[test]
    fn sequence_modules_match_torch() {
        let xs = t(
            &[
                0.5, -1.5, 1.0, 0.25, -0.5, 0.75, //
                1.0, 0.25, 0.0, -1.0, 0.5, 0.5,
            ],
            &[2, 3, 2],
        );
        let h0s = Tensor::zeros(
            &[2, 2, 2],
            crate::dtype::DType::F64,
            &crate::backend::Device::Cpu,
        )
        .unwrap();

        let mut rnn = RNN::new(2, 2, 2).unwrap();
        for cell in &mut rnn.cells {
            cell.ih = lin(&wmat(2), &bias_vec(2, 1.0));
            cell.hh = lin(&wmat(2).mul_scalar(2.0).unwrap(), &bias_vec(2, 3.0));
        }
        let (out, hn) = rnn.forward_seq(&xs, Some(&h0s)).unwrap();
        assert_eq!(out.shape(), &[2, 3, 2]);
        assert_close(
            &out,
            &[
                0.040403828958,
                0.080675956886,
                0.040196171949,
                0.080262660928,
                0.039981248142,
                0.079834880212,
                0.040179067713,
                0.080228617756,
                0.040064183715,
                0.079999956394,
                0.040018871205,
                0.079909766117,
            ],
        );
        assert_close(
            &hn,
            &[
                -0.047055879319,
                -0.093903831536,
                0.06505337653,
                0.129558469159,
                0.039981248142,
                0.079834880212,
                0.040018871205,
                0.079909766117,
            ],
        );

        let mut lstm = LSTM::new(2, 2, 2).unwrap();
        for cell in &mut lstm.cells {
            cell.ih = lin(&wmat(8), &bias_vec(8, 1.0));
            cell.hh = lin(&wmat(8).mul_scalar(2.0).unwrap(), &bias_vec(8, 3.0));
        }
        let c0s = Tensor::zeros(
            &[2, 2, 2],
            crate::dtype::DType::F64,
            &crate::backend::Device::Cpu,
        )
        .unwrap();
        let (out, (hn, cn)) = lstm.forward_seq(&xs, Some((&h0s, &c0s))).unwrap();
        assert_close(
            &out,
            &[
                0.073842457139,
                0.091600318743,
                0.130522575031,
                0.162989170705,
                0.147558079921,
                0.184636690534,
                0.069749083485,
                0.086447143461,
                0.120327319011,
                0.150098740152,
                0.153130483709,
                0.19173657649,
            ],
        );
        assert_close(
            &hn,
            &[
                0.105072575305,
                0.12650570417,
                0.273673678981,
                0.341088797268,
                0.147558079921,
                0.184636690534,
                0.153130483709,
                0.19173657649,
            ],
        );
        assert_close(
            &cn,
            &[
                0.22828948177,
                0.279903103068,
                0.452339653352,
                0.566578280907,
                0.25400336088,
                0.314490212346,
                0.259887299115,
                0.321630930032,
            ],
        );

        let mut gru = GRU::new(2, 2, 2).unwrap();
        for cell in &mut gru.cells {
            cell.ih = lin(&wmat(6), &bias_vec(6, 1.0));
            cell.hh = lin(&wmat(6).mul_scalar(2.0).unwrap(), &bias_vec(6, 3.0));
        }
        let (out, hn) = gru.forward_seq(&xs, Some(&h0s)).unwrap();
        assert_close(
            &out,
            &[
                0.08493159158,
                0.099576344175,
                0.149213806171,
                0.17556873509,
                0.161652115875,
                0.1926694404,
                0.080214089163,
                0.094243711605,
                0.137303494224,
                0.162101786396,
                0.1730516221,
                0.205491446511,
            ],
        );
        assert_close(
            &hn,
            &[
                0.06474545105,
                0.051274966068,
                0.299057115432,
                0.344473705893,
                0.161652115875,
                0.1926694404,
                0.1730516221,
                0.205491446511,
            ],
        );
    }

    /// Wiring invariants (no torch needed): the top layer's final state equals
    /// its last output row, an explicit zero state matches `None`, and a
    /// nonzero state changes the trajectory.
    #[test]
    fn forward_seq_threads_initial_states() {
        let xs = t(&[0.5, -1.5, 1.0, 0.25, -0.5, 0.75], &[1, 3, 2]);
        let mut lstm = LSTM::new(2, 2, 1).unwrap();
        for cell in &mut lstm.cells {
            cell.ih = lin(&wmat(8), &bias_vec(8, 1.0));
            cell.hh = lin(&wmat(8).mul_scalar(2.0).unwrap(), &bias_vec(8, 3.0));
        }
        let (out_none, (hn_none, _)) = lstm.forward_seq(&xs, None).unwrap();
        let zeros = Tensor::zeros(
            &[1, 1, 2],
            crate::dtype::DType::F64,
            &crate::backend::Device::Cpu,
        )
        .unwrap();
        let (out_zero, (hn_zero, _)) = lstm.forward_seq(&xs, Some((&zeros, &zeros))).unwrap();
        assert_close(&out_none, &out_zero.to_vec::<f64>().unwrap());
        assert_close(&hn_none, &hn_zero.to_vec::<f64>().unwrap());
        // h_n[0] is the last output row of the (single) layer.
        let last = out_none.select(1, 2).unwrap();
        assert_close(
            &hn_none.select(0, 0).unwrap(),
            &last.to_vec::<f64>().unwrap(),
        );
        // A nonzero initial state perturbs the trajectory.
        let ones = Tensor::ones(
            &[1, 1, 2],
            crate::dtype::DType::F64,
            &crate::backend::Device::Cpu,
        )
        .unwrap();
        let (out_one, _) = lstm.forward_seq(&xs, Some((&ones, &ones))).unwrap();
        let d: f64 = out_one
            .sub(&out_none)
            .unwrap()
            .abs()
            .unwrap()
            .sum()
            .unwrap()
            .to_vec::<f64>()
            .unwrap()[0];
        assert!(d > 1e-6, "h0/c0 must affect the outputs, got {d}");
    }

    /// Malformed inputs are errors, never panics: non-3D input, zero layers,
    /// empty sequences (the per-step loop would leave `state` as `None`), and
    /// initial states with the wrong layer count.
    #[test]
    fn forward_seq_rejects_bad_input() {
        assert!(RNN::new(2, 2, 0).is_err());
        assert!(LSTM::new(2, 2, 0).is_err());
        assert!(GRU::new(2, 2, 0).is_err());
        let rnn = RNN::new(2, 2, 1).unwrap();
        assert!(rnn.forward_seq(&t(&[1.0, 2.0], &[1, 2]), None).is_err());
        assert!(
            rnn.forward_seq(
                &Tensor::zeros(
                    &[1, 0, 2],
                    crate::dtype::DType::F64,
                    &crate::backend::Device::Cpu
                )
                .unwrap(),
                None
            )
            .is_err()
        );
        // All-F32 so only the layer-count mismatch can fail (torch also rejects
        // h0 whose first dim is not num_layers instead of ignoring extras).
        let bad_h0 = Tensor::zeros(
            &[3, 1, 2],
            crate::dtype::DType::F32,
            &crate::backend::Device::Cpu,
        )
        .unwrap();
        let x32 = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 2, 2]).unwrap();
        assert!(rnn.forward_seq(&x32, Some(&bad_h0)).is_err());
        let cell = RNNCell::new(2, 2).unwrap();
        assert!(cell.step(&t(&[1.0, 2.0], &[2]), None).is_err());
    }
}
