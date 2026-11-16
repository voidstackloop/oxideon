"""Physics-Informed Neural Networks (PINNs) and PDE Residual Solvers for Oxideon.

Enables solving scientific forward and inverse differential equations via autograd:
  - PINNLoss: Joint loss balancing boundary/measurement data and physical PDE residuals
  - heat_equation_residual: Thermal diffusion equation residual (u_t - alpha * u_xx)
  - wave_equation_residual: Hyperbolic acoustic/wave equation residual (u_tt - c^2 * u_xx)
  - burgers_equation_residual: Non-linear viscous Burgers' fluid equation residual (u_t + u * u_x - nu * u_xx)
  - navier_stokes_2d_residual: Incompressible 2D Navier-Stokes momentum residuals
"""

from typing import Tuple, Optional, Any, Callable
import numpy as np

from .oxideon import Tensor as _Tensor, from_numpy as _from_numpy
from .nn import Module


class PINNLoss(Module):
    """Composite loss balancing supervised boundary/initial conditions with PDE conservation laws."""

    def __init__(self, pde_weight: float = 1.0, data_weight: float = 1.0):
        super().__init__()
        if pde_weight < 0.0:
            raise ValueError(f"pde_weight must be >= 0, got {pde_weight}")
        if data_weight < 0.0:
            raise ValueError(f"data_weight must be >= 0, got {data_weight}")
        self.pde_weight = float(pde_weight)
        self.data_weight = float(data_weight)

    def forward(
        self,
        pred_data: Any,
        true_data: Any,
        pde_residual: Any,
    ) -> _Tensor:
        """Computes: L = data_weight * MSE(pred, true) + pde_weight * mean(pde_residual^2)."""
        if not isinstance(pred_data, _Tensor):
            pred_data = _from_numpy(np.ascontiguousarray(pred_data, dtype=np.float32))
        if not isinstance(true_data, _Tensor):
            true_data = _from_numpy(np.ascontiguousarray(true_data, dtype=np.float32))
        if not isinstance(pde_residual, _Tensor):
            pde_residual = _from_numpy(np.ascontiguousarray(pde_residual, dtype=np.float32))

        if pred_data.shape != true_data.shape:
            raise ValueError(f"Data shape mismatch: pred_data {pred_data.shape} != true_data {true_data.shape}")
        diff_data = pred_data - true_data
        l_data = (diff_data * diff_data).mean()

        l_pde = (pde_residual * pde_residual).mean()

        return l_data * self.data_weight + l_pde * self.pde_weight


def heat_equation_residual(
    u_t: Any, u_xx: Any, alpha: float = 0.01
) -> _Tensor:
    """Computes Heat equation residual: R = u_t - alpha * u_xx."""
    if alpha < 0.0:
        raise ValueError(f"alpha must be >= 0, got {alpha}")
    if isinstance(u_t, _Tensor) or isinstance(u_xx, _Tensor):
        t_ut = u_t if isinstance(u_t, _Tensor) else _from_numpy(np.ascontiguousarray(u_t, dtype=np.float32))
        t_uxx = u_xx if isinstance(u_xx, _Tensor) else _from_numpy(np.ascontiguousarray(u_xx, dtype=np.float32))
        if t_ut.shape != t_uxx.shape:
            raise ValueError(f"Shape mismatch: u_t shape {t_ut.shape} != u_xx shape {t_uxx.shape}")
        return t_ut - (t_uxx * float(alpha))

    ut = np.asarray(u_t, dtype=np.float32)
    uxx = np.asarray(u_xx, dtype=np.float32)
    if ut.shape != uxx.shape:
        raise ValueError(f"Shape mismatch: u_t shape {ut.shape} != u_xx shape {uxx.shape}")
    res = ut - alpha * uxx
    return _from_numpy(np.ascontiguousarray(res, dtype=np.float32))


def wave_equation_residual(
    u_tt: Any, u_xx: Any, c: float = 1.0
) -> _Tensor:
    """Computes Wave equation residual: R = u_tt - c^2 * u_xx."""
    if c <= 0.0:
        raise ValueError(f"c must be > 0, got {c}")
    if isinstance(u_tt, _Tensor) or isinstance(u_xx, _Tensor):
        t_utt = u_tt if isinstance(u_tt, _Tensor) else _from_numpy(np.ascontiguousarray(u_tt, dtype=np.float32))
        t_uxx = u_xx if isinstance(u_xx, _Tensor) else _from_numpy(np.ascontiguousarray(u_xx, dtype=np.float32))
        if t_utt.shape != t_uxx.shape:
            raise ValueError(f"Shape mismatch: u_tt shape {t_utt.shape} != u_xx shape {t_uxx.shape}")
        return t_utt - (t_uxx * float(c ** 2))

    utt = np.asarray(u_tt, dtype=np.float32)
    uxx = np.asarray(u_xx, dtype=np.float32)
    if utt.shape != uxx.shape:
        raise ValueError(f"Shape mismatch: u_tt shape {utt.shape} != u_xx shape {uxx.shape}")
    res = utt - (c**2) * uxx
    return _from_numpy(np.ascontiguousarray(res, dtype=np.float32))


def burgers_equation_residual(
    u: Any, u_t: Any, u_x: Any, u_xx: Any, nu: float = 0.01
) -> _Tensor:
    """Computes Burgers' equation residual: R = u_t + u * u_x - nu * u_xx."""
    if nu < 0.0:
        raise ValueError(f"nu must be >= 0, got {nu}")
    if any(isinstance(x, _Tensor) for x in (u, u_t, u_x, u_xx)):
        t_u = u if isinstance(u, _Tensor) else _from_numpy(np.ascontiguousarray(u, dtype=np.float32))
        t_ut = u_t if isinstance(u_t, _Tensor) else _from_numpy(np.ascontiguousarray(u_t, dtype=np.float32))
        t_ux = u_x if isinstance(u_x, _Tensor) else _from_numpy(np.ascontiguousarray(u_x, dtype=np.float32))
        t_uxx = u_xx if isinstance(u_xx, _Tensor) else _from_numpy(np.ascontiguousarray(u_xx, dtype=np.float32))
        if not (t_u.shape == t_ut.shape == t_ux.shape == t_uxx.shape):
            raise ValueError(
                f"Shape mismatch across Burgers' inputs: u={t_u.shape}, u_t={t_ut.shape}, u_x={t_ux.shape}, u_xx={t_uxx.shape}"
            )
        return t_ut + (t_u * t_ux) - (t_uxx * float(nu))

    u_arr = np.asarray(u, dtype=np.float32)
    ut = np.asarray(u_t, dtype=np.float32)
    ux = np.asarray(u_x, dtype=np.float32)
    uxx = np.asarray(u_xx, dtype=np.float32)

    if not (u_arr.shape == ut.shape == ux.shape == uxx.shape):
        raise ValueError(
            f"Shape mismatch across Burgers' inputs: u={u_arr.shape}, u_t={ut.shape}, u_x={ux.shape}, u_xx={uxx.shape}"
        )

    res = ut + u_arr * ux - nu * uxx
    return _from_numpy(np.ascontiguousarray(res, dtype=np.float32))


def navier_stokes_2d_residual(
    u: Any,
    v: Any,
    u_t: Any,
    v_t: Any,
    u_x: Any,
    u_y: Any,
    v_x: Any,
    v_y: Any,
    u_xx: Any,
    u_yy: Any,
    v_xx: Any,
    v_yy: Any,
    p_x: Any,
    p_y: Any,
    rho: float = 1.0,
    nu: float = 0.01,
) -> Tuple[_Tensor, _Tensor]:
    """Computes 2D incompressible Navier-Stokes momentum residuals (res_u, res_v)."""
    if rho <= 0.0:
        raise ValueError(f"rho must be > 0, got {rho}")
    if nu < 0.0:
        raise ValueError(f"nu must be >= 0, got {nu}")

    def _to_t(val):
        return val if isinstance(val, _Tensor) else _from_numpy(np.ascontiguousarray(val, dtype=np.float32))

    t_u = _to_t(u)
    t_v = _to_t(v)
    t_ut = _to_t(u_t)
    t_vt = _to_t(v_t)
    t_ux = _to_t(u_x)
    t_uy = _to_t(u_y)
    t_vx = _to_t(v_x)
    t_vy = _to_t(v_y)
    t_uxx = _to_t(u_xx)
    t_uyy = _to_t(u_yy)
    t_vxx = _to_t(v_xx)
    t_vyy = _to_t(v_yy)
    t_px = _to_t(p_x)
    t_py = _to_t(p_y)

    inv_rho = 1.0 / float(rho)
    nu_f = float(nu)

    res_u = t_ut + (t_u * t_ux) + (t_v * t_uy) + (t_px * inv_rho) - ((t_uxx + t_uyy) * nu_f)
    res_v = t_vt + (t_u * t_vx) + (t_v * t_vy) + (t_py * inv_rho) - ((t_vxx + t_vyy) * nu_f)

    return res_u, res_v


__all__ = [
    "PINNLoss",
    "heat_equation_residual",
    "wave_equation_residual",
    "burgers_equation_residual",
    "navier_stokes_2d_residual",
]
