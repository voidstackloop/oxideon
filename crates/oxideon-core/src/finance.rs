//! Quantitative-finance calculators (numerical utilities, not investment advice): Black-Scholes pricing with
//! Greeks and [`implied_volatility`], Cox-Ross-Rubinstein [`binomial_price`] (European / American), cash-flow
//! maths ([`npv`], [`irr`], [`loan_payment`]), bond pricing with duration, and return-series risk measures
//! ([`value_at_risk`], [`expected_shortfall`], [`max_drawdown`], [`sharpe_ratio`]).

use crate::error::{Error, Result};
use crate::stats_tests::norm_cdf;

fn norm_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

/// Option inputs: spot `s`, strike `k`, risk-free rate `r`, dividend yield `q`, volatility `sigma`, expiry `t` (years).
#[derive(Debug, Clone, Copy)]
pub struct OptionSpec {
    pub s: f64,
    pub k: f64,
    pub r: f64,
    pub q: f64,
    pub sigma: f64,
    pub t: f64,
}

impl OptionSpec {
    fn check(&self) -> Result<()> {
        if self.s <= 0.0 || self.k <= 0.0 || self.sigma <= 0.0 || self.t <= 0.0 || [self.s, self.k, self.r, self.q, self.sigma, self.t].iter().any(|v| !v.is_finite()) {
            return Err(Error::Msg("OptionSpec: need positive finite spot, strike, volatility and expiry".into()));
        }
        Ok(())
    }

    fn d1d2(&self) -> (f64, f64) {
        let st = self.sigma * self.t.sqrt();
        let d1 = ((self.s / self.k).ln() + (self.r - self.q + 0.5 * self.sigma * self.sigma) * self.t) / st;
        (d1, d1 - st)
    }
}

/// Greeks of a European option.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Greeks {
    pub price: f64,
    pub delta: f64,
    pub gamma: f64,
    /// Per 1.00 (100 percentage points) of volatility.
    pub vega: f64,
    /// Per year.
    pub theta: f64,
    /// Per 1.00 of the interest rate.
    pub rho: f64,
}

/// Black-Scholes-Merton price and Greeks of a European call (`is_call`) or put.
pub fn black_scholes(o: &OptionSpec, is_call: bool) -> Result<Greeks> {
    o.check()?;
    let (d1, d2) = o.d1d2();
    let (eq, er) = ((-o.q * o.t).exp(), (-o.r * o.t).exp());
    let sign = if is_call { 1.0 } else { -1.0 };
    let price = sign * (o.s * eq * norm_cdf(sign * d1) - o.k * er * norm_cdf(sign * d2));
    let delta = sign * eq * norm_cdf(sign * d1);
    let gamma = eq * norm_pdf(d1) / (o.s * o.sigma * o.t.sqrt());
    let vega = o.s * eq * norm_pdf(d1) * o.t.sqrt();
    let theta = -o.s * eq * norm_pdf(d1) * o.sigma / (2.0 * o.t.sqrt()) - sign * o.r * o.k * er * norm_cdf(sign * d2) + sign * o.q * o.s * eq * norm_cdf(sign * d1);
    let rho = sign * o.k * o.t * er * norm_cdf(sign * d2);
    Ok(Greeks { price, delta, gamma, vega, theta, rho })
}

/// Volatility that reproduces `market_price` (bisection on `[1e-4, 5]`); errors if the price is outside the
/// no-arbitrage range for those bounds.
pub fn implied_volatility(market_price: f64, o: &OptionSpec, is_call: bool) -> Result<f64> {
    let price_at = |sigma: f64| black_scholes(&OptionSpec { sigma, ..*o }, is_call).map(|g| g.price);
    let (mut lo, mut hi) = (1e-4, 5.0);
    let (plo, phi) = (price_at(lo)?, price_at(hi)?);
    if market_price < plo - 1e-12 || market_price > phi + 1e-12 {
        return Err(Error::Msg("implied_volatility: the price is outside the attainable range".into()));
    }
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if price_at(mid)? < market_price { lo = mid } else { hi = mid }
        if hi - lo < 1e-12 {
            break;
        }
    }
    Ok(0.5 * (lo + hi))
}

/// Cox-Ross-Rubinstein binomial tree with `steps` steps; `american` allows early exercise.
pub fn binomial_price(o: &OptionSpec, is_call: bool, american: bool, steps: usize) -> Result<f64> {
    o.check()?;
    if steps == 0 {
        return Err(Error::Msg("binomial_price: steps must be positive".into()));
    }
    let dt = o.t / steps as f64;
    let (u, disc) = ((o.sigma * dt.sqrt()).exp(), (-o.r * dt).exp());
    let d = 1.0 / u;
    let p = (((o.r - o.q) * dt).exp() - d) / (u - d);
    if !(0.0..=1.0).contains(&p) {
        return Err(Error::Msg("binomial_price: risk-neutral probability outside [0, 1]; use more steps".into()));
    }
    let payoff = |s: f64| if is_call { (s - o.k).max(0.0) } else { (o.k - s).max(0.0) };
    let mut v: Vec<f64> = (0..=steps).map(|j| payoff(o.s * u.powi(j as i32) * d.powi((steps - j) as i32))).collect();
    for i in (0..steps).rev() {
        for j in 0..=i {
            let cont = disc * (p * v[j + 1] + (1.0 - p) * v[j]);
            v[j] = if american { cont.max(payoff(o.s * u.powi(j as i32) * d.powi((i - j) as i32))) } else { cont };
        }
    }
    Ok(v[0])
}

/// Net present value of `cash_flows[t]` received at the end of period `t` (`t = 0` is undiscounted).
pub fn npv(rate: f64, cash_flows: &[f64]) -> f64 {
    cash_flows.iter().enumerate().map(|(t, c)| c / (1.0 + rate).powi(t as i32)).sum()
}

/// Internal rate of return (bisection on `(-0.99, 10)`); needs a sign change in the cash flows.
pub fn irr(cash_flows: &[f64]) -> Result<f64> {
    if !(cash_flows.iter().any(|&c| c > 0.0) && cash_flows.iter().any(|&c| c < 0.0)) {
        return Err(Error::Msg("irr: the cash flows need both positive and negative entries".into()));
    }
    let (mut lo, mut hi) = (-0.99, 10.0);
    let (flo, fhi) = (npv(lo, cash_flows), npv(hi, cash_flows));
    if flo * fhi > 0.0 {
        return Err(Error::Msg("irr: no root found in (-99%, 1000%)".into()));
    }
    for _ in 0..300 {
        let mid = 0.5 * (lo + hi);
        if npv(mid, cash_flows) * flo > 0.0 { lo = mid } else { hi = mid }
    }
    Ok(0.5 * (lo + hi))
}

/// Level payment of a fully amortizing loan: `principal` over `periods` at periodic `rate`.
pub fn loan_payment(principal: f64, rate: f64, periods: u32) -> Result<f64> {
    if periods == 0 || principal < 0.0 || rate <= -1.0 {
        return Err(Error::Msg("loan_payment: need periods >= 1, principal >= 0 and rate > -1".into()));
    }
    Ok(if rate == 0.0 { principal / periods as f64 } else { principal * rate / (1.0 - (1.0 + rate).powi(-(periods as i32))) })
}

/// Price, Macaulay duration (years) and modified duration of a bond paying `coupon_rate * face / freq` per period.
pub fn bond(face: f64, coupon_rate: f64, yield_rate: f64, years: f64, freq: u32) -> Result<(f64, f64, f64)> {
    if face <= 0.0 || years <= 0.0 || freq == 0 || yield_rate <= -1.0 {
        return Err(Error::Msg("bond: need positive face, years and frequency, and yield > -100%".into()));
    }
    let n = (years * freq as f64).round() as u32;
    let (c, y) = (face * coupon_rate / freq as f64, yield_rate / freq as f64);
    let flows: Vec<f64> = (1..=n).map(|k| c + if k == n { face } else { 0.0 }).collect();
    let pv: Vec<f64> = flows.iter().enumerate().map(|(i, f)| f / (1.0 + y).powi(i as i32 + 1)).collect();
    let price: f64 = pv.iter().sum();
    let mac = pv.iter().enumerate().map(|(i, v)| (i as f64 + 1.0) / freq as f64 * v).sum::<f64>() / price;
    Ok((price, mac, mac / (1.0 + y)))
}

fn sorted(returns: &[f64]) -> Result<Vec<f64>> {
    if returns.is_empty() || returns.iter().any(|v| !v.is_finite()) {
        return Err(Error::Msg("returns must be a non-empty finite series".into()));
    }
    let mut s = returns.to_vec();
    s.sort_by(f64::total_cmp);
    Ok(s)
}

/// Historical value at risk: the loss (positive number) exceeded with probability `1 - confidence`.
pub fn value_at_risk(returns: &[f64], confidence: f64) -> Result<f64> {
    if !(0.0 < confidence && confidence < 1.0) {
        return Err(Error::Msg("value_at_risk: confidence must lie in (0, 1)".into()));
    }
    let s = sorted(returns)?;
    let idx = ((((1.0 - confidence) * s.len() as f64) - 1e-9).ceil() as usize).clamp(1, s.len()) - 1;
    Ok(-s[idx])
}

/// Expected shortfall (CVaR): the mean loss in the worst `1 - confidence` tail.
pub fn expected_shortfall(returns: &[f64], confidence: f64) -> Result<f64> {
    if !(0.0 < confidence && confidence < 1.0) {
        return Err(Error::Msg("expected_shortfall: confidence must lie in (0, 1)".into()));
    }
    let s = sorted(returns)?;
    let k = ((((1.0 - confidence) * s.len() as f64) - 1e-9).ceil() as usize).clamp(1, s.len());
    Ok(-s[..k].iter().sum::<f64>() / k as f64)
}

/// Largest peak-to-trough decline of the cumulative wealth curve built from simple `returns` (as a fraction).
pub fn max_drawdown(returns: &[f64]) -> Result<f64> {
    sorted(returns)?;
    let (mut wealth, mut peak, mut worst) = (1.0f64, 1.0f64, 0.0f64);
    for r in returns {
        wealth *= 1.0 + r;
        peak = peak.max(wealth);
        worst = worst.max(1.0 - wealth / peak);
    }
    Ok(worst)
}

/// Annualized Sharpe ratio from per-period `returns`, a per-period risk-free rate and `periods_per_year`.
pub fn sharpe_ratio(returns: &[f64], risk_free: f64, periods_per_year: f64) -> Result<f64> {
    sorted(returns)?;
    if returns.len() < 2 {
        return Err(Error::Msg("sharpe_ratio: need at least two returns".into()));
    }
    let ex: Vec<f64> = returns.iter().map(|r| r - risk_free).collect();
    let m = ex.iter().sum::<f64>() / ex.len() as f64;
    let sd = (ex.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (ex.len() as f64 - 1.0)).sqrt();
    if sd == 0.0 {
        return Err(Error::Msg("sharpe_ratio: zero volatility".into()));
    }
    Ok(m / sd * periods_per_year.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() < tol, "{a} vs {b}");
    }

    fn spec() -> OptionSpec {
        OptionSpec { s: 100.0, k: 100.0, r: 0.05, q: 0.0, sigma: 0.2, t: 1.0 }
    }

    #[test]
    fn black_scholes_matches_reference_values() {
        // textbook: S=K=100, r=5%, sigma=20%, T=1 -> call 10.4506, put 5.5735
        let call = black_scholes(&spec(), true).unwrap();
        let put = black_scholes(&spec(), false).unwrap();
        close(call.price, 10.4506, 1e-3);
        close(put.price, 5.5735, 1e-3);
        // put-call parity: C - P = S e^{-qT} - K e^{-rT}
        close(call.price - put.price, 100.0 - 100.0 * (-0.05f64).exp(), 1e-9);
        close(call.delta, 0.6368, 1e-3);
        close(call.delta - put.delta, 1.0, 1e-9);
        close(call.gamma, put.gamma, 1e-12);
        close(call.vega, 37.524, 5e-3);
        // Greeks agree with finite differences of the price
        let bump = |f: &dyn Fn(f64) -> OptionSpec| -> f64 { (black_scholes(&f(1e-4), true).unwrap().price - black_scholes(&f(-1e-4), true).unwrap().price) / 2e-4 };
        close(bump(&|h| OptionSpec { s: 100.0 + h * 100.0, ..spec() }) / 100.0, call.delta, 1e-5);
        close(bump(&|h| OptionSpec { sigma: 0.2 + h, ..spec() }), call.vega, 1e-3);
        close(bump(&|h| OptionSpec { r: 0.05 + h, ..spec() }), call.rho, 1e-3);
        close(-bump(&|h| OptionSpec { t: 1.0 + h, ..spec() }), call.theta, 1e-3);
        assert!(black_scholes(&OptionSpec { sigma: 0.0, ..spec() }, true).is_err());
    }

    #[test]
    fn implied_vol_and_binomial_trees() {
        let price = black_scholes(&OptionSpec { sigma: 0.37, ..spec() }, true).unwrap().price;
        close(implied_volatility(price, &spec(), true).unwrap(), 0.37, 1e-8);
        assert!(implied_volatility(1000.0, &spec(), true).is_err());
        // European tree converges to Black-Scholes; American put is worth at least as much (early exercise)
        let eu = binomial_price(&spec(), false, false, 800).unwrap();
        close(eu, 5.5735, 0.01);
        let am = binomial_price(&spec(), false, true, 800).unwrap();
        assert!(am > eu + 0.1, "{am} vs {eu}");
        // an American call on a non-dividend stock is never exercised early
        close(binomial_price(&spec(), true, true, 400).unwrap(), binomial_price(&spec(), true, false, 400).unwrap(), 1e-9);
        assert!(binomial_price(&spec(), true, false, 0).is_err());
    }

    #[test]
    fn cash_flow_and_bond_maths() {
        close(npv(0.1, &[-100.0, 60.0, 60.0]), -100.0 + 60.0 / 1.1 + 60.0 / 1.21, 1e-12);
        close(irr(&[-100.0, 60.0, 60.0]).unwrap(), 0.130_662_386, 1e-8);
        assert!(npv(irr(&[-1000.0, 300.0, 400.0, 500.0]).unwrap(), &[-1000.0, 300.0, 400.0, 500.0]).abs() < 1e-8);
        assert!(irr(&[100.0, 50.0]).is_err());
        // 30-year 6% mortgage on 200,000 -> 1,199.10 per month
        close(loan_payment(200_000.0, 0.06 / 12.0, 360).unwrap(), 1199.10, 0.01);
        close(loan_payment(1200.0, 0.0, 12).unwrap(), 100.0, 1e-12);
        // a bond priced at par has price == face; zero-coupon duration equals maturity
        let (p, mac, modd) = bond(1000.0, 0.05, 0.05, 10.0, 2).unwrap();
        close(p, 1000.0, 1e-9);
        assert!(mac > 7.0 && mac < 8.5 && modd < mac);
        let (zp, zmac, _) = bond(1000.0, 0.0, 0.04, 5.0, 1).unwrap();
        close(zp, 1000.0 / 1.04f64.powi(5), 1e-9);
        close(zmac, 5.0, 1e-9);
        assert!(bond(0.0, 0.05, 0.05, 5.0, 2).is_err());
    }

    #[test]
    fn risk_measures() {
        let r: Vec<f64> = (0..100).map(|i| (i as f64 - 50.0) / 1000.0).collect(); // -5.0% .. +4.9%
        close(value_at_risk(&r, 0.95).unwrap(), 0.046, 1e-12);
        let es = expected_shortfall(&r, 0.95).unwrap();
        assert!(es > value_at_risk(&r, 0.95).unwrap() && (es - 0.048).abs() < 1e-12, "{es}");
        // +10%, -20%, +5%: wealth 1.1, 0.88, 0.924 -> drawdown 20%
        close(max_drawdown(&[0.10, -0.20, 0.05]).unwrap(), 0.20, 1e-12);
        close(max_drawdown(&[0.01, 0.02]).unwrap(), 0.0, 1e-12);
        let s = sharpe_ratio(&[0.02, 0.01, 0.03, 0.00, 0.02], 0.0, 12.0).unwrap();
        close(s, 0.016 / (0.011_401_754_250_991_38) * 12f64.sqrt(), 1e-9);
        assert!(sharpe_ratio(&[0.01, 0.01], 0.0, 12.0).is_err() && value_at_risk(&[], 0.9).is_err() && value_at_risk(&r, 1.0).is_err());
    }
}
