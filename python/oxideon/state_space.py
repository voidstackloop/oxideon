"""
State Space Models (SSMs) and Selective State Space Architectures (Mamba) for Oxideon.
Includes continuous-to-discrete ZOH discretization, selective recurrent scan, and Mamba blocks.
"""
from typing import Optional, Tuple, List
import numpy as np
from .oxideon import Tensor, from_numpy as _from_numpy
from .nn import Module, Linear, Parameter


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))


def _silu_tensor(x: Tensor) -> Tensor:
    return x * x.clamp(-20.0, 20.0).sigmoid()


class SelectiveScan:
    """
    Selective Scan mechanism as introduced in Mamba (Gu & Dao, 2023).
    Performs input-dependent discretization:
      Delta_t = softplus(Parameter + proj(x_t))
      A_bar_t = exp(Delta_t * A)
      B_bar_t = Delta_t * B_t
      h_t = A_bar_t * h_{t-1} + B_bar_t * x_t
      y_t = C_t * h_t
    """
    @staticmethod
    def forward(
        x: np.ndarray,      # (B, L, D) or (L, D)
        delta: np.ndarray,  # (B, L, D) or (L, D)
        A: np.ndarray,      # (D, N)
        B: np.ndarray,      # (B, L, N) or (L, N)
        C: np.ndarray,      # (B, L, N) or (L, N)
        D_param: np.ndarray # (D,)
    ) -> np.ndarray:
        is_2d = x.ndim == 2
        if is_2d:
            x = x[None, :, :]
            delta = delta[None, :, :]
            B = B[None, :, :]
            C = C[None, :, :]
        batch_size, seq_len, d_inner = x.shape
        d_state = A.shape[1]

        y = np.zeros_like(x)
        h = np.zeros((batch_size, d_inner, d_state), dtype=np.float32)

        for t in range(seq_len):
            xt = x[:, t, :]
            dt = delta[:, t, :]
            bt = B[:, t, :]
            ct = C[:, t, :]

            a_bar = np.exp(dt[:, :, None] * A[None, :, :])
            b_bar = dt[:, :, None] * bt[:, None, :]

            h = a_bar * h + b_bar * xt[:, :, None]
            y[:, t, :] = np.einsum('bdn,bn->bd', h, ct)

        y += x * D_param[None, None, :]
        return y[0] if is_2d else y

    @staticmethod
    def forward_tensor(
        x: Tensor,      # (B, L, D)
        delta: Tensor,  # (B, L, D)
        A: Tensor,      # (D, N)
        B: Tensor,      # (B, L, N)
        C: Tensor,      # (B, L, N)
        D_param: Tensor # (D,)
    ) -> Tensor:
        batch_size, seq_len, d_inner = x.shape
        d_state = A.shape[1]

        h: Optional[Tensor] = None
        A_expanded = A.unsqueeze(0)  # (1, D, N)
        y_steps: List[Tensor] = []

        for t in range(seq_len):
            # Extract timestep t via one-hot selector along L
            sel_np = np.zeros((1, seq_len, 1), dtype=np.float32)
            sel_np[0, t, 0] = 1.0
            sel_t = _from_numpy(sel_np)

            xt = (x * sel_t).sum(dim=1)        # (B, D)
            dt = (delta * sel_t).sum(dim=1)    # (B, D)
            bt = (B * sel_t).sum(dim=1)        # (B, N)
            ct = (C * sel_t).sum(dim=1)        # (B, N)

            dt_3d = dt.unsqueeze(2)            # (B, D, 1)
            bt_3d = bt.unsqueeze(1)            # (B, 1, N)
            ct_3d = ct.unsqueeze(1)            # (B, 1, N)
            xt_3d = xt.unsqueeze(2)            # (B, D, 1)

            a_bar = (dt_3d * A_expanded).clamp(-20.0, 5.0).exp()  # (B, D, N)
            b_bar = dt_3d * bt_3d                                  # (B, D, N)

            if h is None:
                h = b_bar * xt_3d
            else:
                h = a_bar * h + b_bar * xt_3d

            yt = (h * ct_3d).sum(dim=-1)       # (B, D)
            y_steps.append(yt.unsqueeze(1))    # (B, 1, D)

        y = Tensor.cat(y_steps, dim=1)         # (B, L, D)
        return y + x * D_param.reshape([1, 1, d_inner])


class MambaBlock(Module):
    """
    Full Mamba selective state-space block combining gated projection,
    depthwise 1D causal convolution, and input-dependent selective scan.
    """
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        if d_model <= 0 or d_state <= 0 or d_conv <= 0 or expand <= 0:
            raise ValueError(f"d_model, d_state, d_conv, and expand must be positive, got {d_model}, {d_state}, {d_conv}, {expand}")
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = d_model * expand

        # Projections
        self.in_proj = Linear(d_model, 2 * self.d_inner)
        self.x_proj = Linear(self.d_inner, self.d_inner + 2 * d_state)
        self.out_proj = Linear(self.d_inner, d_model)

        # Diagonal HiPPO continuous state transition matrix A
        a_diag = -np.arange(1, d_state + 1, dtype=np.float32)[None, :]
        self.A = Parameter(_from_numpy(np.repeat(a_diag, self.d_inner, axis=0)))
        self.D = Parameter(_from_numpy(np.ones(self.d_inner, dtype=np.float32)))

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor of shape (B, L, D) or (L, D)
        Returns:
            out: Tensor of shape (B, L, D) or (L, D)
        """
        orig_shape = list(x.shape)
        single = False
        if len(orig_shape) == 2:
            x_3d = x.unsqueeze(0)
            single = True
        elif len(orig_shape) == 3:
            x_3d = x
        else:
            raise ValueError(f"MambaBlock requires 2D or 3D input tensor, got shape {orig_shape}")

        B, L, D = x_3d.shape
        if D != self.d_model:
            raise ValueError(f"Feature dimension mismatch: expected {self.d_model}, got {D}")
        if L == 0 or B == 0:
            return x

        projected = self.in_proj(x_3d)  # (B, L, 2 * d_inner)

        # Split projected into x_branch and z_branch via projection masks
        mask_x_np = np.zeros((2 * self.d_inner, self.d_inner), dtype=np.float32)
        mask_x_np[:self.d_inner, :] = np.eye(self.d_inner, dtype=np.float32)
        mask_z_np = np.zeros((2 * self.d_inner, self.d_inner), dtype=np.float32)
        mask_z_np[self.d_inner:, :] = np.eye(self.d_inner, dtype=np.float32)

        x_branch = projected.matmul(_from_numpy(mask_x_np))
        z_branch = projected.matmul(_from_numpy(mask_z_np))

        # Causal depthwise 1D moving-window mixing across [t - d_conv + 1 .. t]
        k_win = min(self.d_conv, L)
        causal_mat = np.zeros((L, L), dtype=np.float32)
        for i in range(L):
            start = max(0, i - k_win + 1)
            causal_mat[start : i + 1, i] = 1.0 / float(i - start + 1)
        x_conv = x_branch.transpose(1, 2).matmul(_from_numpy(causal_mat)).transpose(1, 2)
        x_act = _silu_tensor(x_conv)

        # Dynamic delta, B, C from x_act
        ssm_params = self.x_proj(x_act)  # (B, L, d_inner + 2 * d_state)
        total_ssm = self.d_inner + 2 * self.d_state

        sel_delta = np.zeros((total_ssm, self.d_inner), dtype=np.float32)
        sel_delta[:self.d_inner, :] = np.eye(self.d_inner, dtype=np.float32)
        sel_B = np.zeros((total_ssm, self.d_state), dtype=np.float32)
        sel_B[self.d_inner : self.d_inner + self.d_state, :] = np.eye(self.d_state, dtype=np.float32)
        sel_C = np.zeros((total_ssm, self.d_state), dtype=np.float32)
        sel_C[self.d_inner + self.d_state :, :] = np.eye(self.d_state, dtype=np.float32)

        delta_raw = ssm_params.matmul(_from_numpy(sel_delta))
        delta = (delta_raw.clamp(-20.0, 20.0).exp() + 1.0).log()
        B_mat = ssm_params.matmul(_from_numpy(sel_B))
        C_mat = ssm_params.matmul(_from_numpy(sel_C))

        y = SelectiveScan.forward_tensor(
            x=x_act,
            delta=delta,
            A=self.A,
            B=B_mat,
            C=C_mat,
            D_param=self.D,
        )

        y_gated = y * _silu_tensor(z_branch)
        out = self.out_proj(y_gated)

        if single:
            return out.squeeze(0)
        return out


__all__ = [
    "SelectiveScan",
    "MambaBlock",
]
