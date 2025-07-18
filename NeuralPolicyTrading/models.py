import torch
import torch.nn as nn
import torch.nn.functional as F

class InputFeature(nn.Module):
    """
    Takes raw price tensor windows and computes simple moving averages.
    Expects input x of shape (batch, seq_len).
    Computes:
        m_short = x[:, -window_short:].mean(dim=1, keepdim=True)
        m_long  = x[:, -window_long:].mean(dim=1, keepdim=True)
    Returns tensor of shape (batch, 2).
    """
    def __init__(self, window_short: int, window_long: int):
        super().__init__()
        self.window_short = window_short
        self.window_long = window_long

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len)
        m_short = x[:, -self.window_short:].mean(dim=1, keepdim=True)
        m_long  = x[:, -self.window_long:].mean(dim=1, keepdim=True)
        return torch.cat([m_short, m_long], dim=1)


class FeatureNet(nn.Module):
    """
    Two sigmoid neurons computing:
      o1 = σ((m_short - m_long) + τ1)
      o2 = σ((m_long  - m_short) + τ2)
    τ1, τ2 are trainable scalars.
    Input: (batch, 2), Output: (batch, 2)
    """
    def __init__(self, tau_init: float = 0.0):
        super().__init__()
        self.tau1 = nn.Parameter(torch.tensor(tau_init, dtype=torch.float32))
        self.tau2 = nn.Parameter(torch.tensor(tau_init, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch,2) => x[:,0]=m_short, x[:,1]=m_long
        diff = x[:, 0:1] - x[:, 1:2]
        o1 = torch.sigmoid(diff + self.tau1)
        o2 = torch.sigmoid(-diff + self.tau2)
        return torch.cat([o1, o2], dim=1)


class LogicNet(nn.Module):
    """
    Implements three non-trainable sigmoid "logic" neurons per Eq.(8)-(10):
      NAND: weights=[-2,-2], bias=3.0
      AND:  weights=[2,2],   bias=-1.5
      NOR:  weights=[-2,-2], bias=1.0
    Input: (batch,2), Output: logit scores (batch,3)
    """
    def __init__(self):
        super().__init__()
        w_nand, b_nand = torch.tensor([-2.0, -2.0]), torch.tensor(3.0)
        w_and,  b_and  = torch.tensor([ 2.0,  2.0]), torch.tensor(-1.5)
        w_nor,  b_nor  = torch.tensor([-2.0, -2.0]), torch.tensor(1.0)
        self.register_buffer('w_nand', w_nand)
        self.register_buffer('b_nand', b_nand)
        self.register_buffer('w_and',  w_and)
        self.register_buffer('b_and',  b_and)
        self.register_buffer('w_nor',  w_nor)
        self.register_buffer('b_nor',  b_nor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch,2)
        l_nand = torch.sigmoid(x @ self.w_nand + self.b_nand)
        l_and  = torch.sigmoid(x @ self.w_and  + self.b_and)
        l_nor  = torch.sigmoid(x @ self.w_nor  + self.b_nor)
        return torch.stack([l_nand, l_and, l_nor], dim=1)


class PolicyNet(nn.Module):
    """
    Applies softargmax (softmax(logits * β)) over logic outputs.
    Input: logits (batch,3), Output: probabilities (batch,3)
    """
    def __init__(self, beta: float = 10.0):
        super().__init__()
        self.beta = beta

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return F.softmax(logits * self.beta, dim=-1)


class TradingPolicy(nn.Module):
    def __init__(self, window_short, window_long, beta=10.0):
        super().__init__()
        self.input_f = InputFeature(window_short, window_long)
        self.feat_n  = FeatureNet(tau_init=0.0)
        self.logic   = LogicNet()
        self.policy  = PolicyNet(beta)

    def forward(self, x):
        # x: (batch, seq_len, 1) or (batch, seq_len)
        # squeeze last dim if needed
        if x.ndim == 3:
            x = x.squeeze(-1)
        feat = self.input_f(x)            # ➔ (batch,2)
        f_out = self.feat_n(feat)         # ➔ (batch,2)
        l_out = self.logic(f_out)         # ➔ (batch,3)
        return self.policy(l_out)         # ➔ (batch,3)
