import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── 1. InputFeature ────────────────────────────────────────────────────
class InputFeature(nn.Module):
    def __init__(self, window_short: int, window_long: int, trainable: bool = False):
        super().__init__()
        self.window_short = window_short
        self.window_long  = window_long

        # two linear layers for SMA
        self.lin_short = nn.Linear(window_short, 1, bias=True)
        self.lin_long  = nn.Linear(window_long,  1, bias=True)

        # init to exact SMA
        nn.init.constant_(self.lin_short.weight, 1.0 / window_short)
        nn.init.constant_(self.lin_short.bias,   0.0)
        nn.init.constant_(self.lin_long.weight,  1.0 / window_long)
        nn.init.constant_(self.lin_long.bias,    0.0)

        # if trainable, perturb with N(0,0.05)
        if trainable:
            nn.init.normal_(self.lin_short.weight, mean=0.0, std=0.05)
            nn.init.normal_(self.lin_short.bias,   mean=0.0, std=0.05)
            nn.init.normal_(self.lin_long.weight,  mean=0.0, std=0.05)
            nn.init.normal_(self.lin_long.bias,    mean=0.0, std=0.05)

        # set requires_grad
        for p in [self.lin_short.weight, self.lin_short.bias,
                  self.lin_long.weight,  self.lin_long.bias]:
            p.requires_grad = trainable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len)
        # apply each SMA linear to the last window
        m_short = self.lin_short(x[:, -self.window_short :])
        m_long  = self.lin_long(x[:, -self.window_long  :])
        return torch.cat([m_short, m_long], dim=1)  # (batch,2)


# ─── 2. FeatureNet ─────────────────────────────────────────────────────
class FeatureNet(nn.Module):
    def __init__(self, tau_init: float = 0.0, trainable: bool = True):
        super().__init__()
        self.tau1 = nn.Parameter(torch.empty(1), requires_grad=trainable)
        self.tau2 = nn.Parameter(torch.empty(1), requires_grad=trainable)
        nn.init.normal_(self.tau1, 0.0, 0.05)
        nn.init.normal_(self.tau2, 0.0, 0.05)

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        # m: (batch,2)
        w1 = torch.tensor([1.0, -1.0], device=m.device)
        w2 = torch.tensor([-1.0, 1.0], device=m.device)
        delta1 = m @ w1 + self.tau1
        delta2 = m @ w2 + self.tau2
        o11 = torch.sigmoid(delta1)
        o12 = torch.sigmoid(delta2)
        return torch.stack([o11, o12], dim=1)  # (batch,2)


# ─── 3. LogicNet ───────────────────────────────────────────────────────
class LogicNet(nn.Module):
    def __init__(self, trainable: bool = False):
        super().__init__()
        self.w_nand = nn.Parameter(torch.tensor([-2.0, -2.0]), requires_grad=trainable)
        self.b_nand = nn.Parameter(torch.tensor( 3.0     ), requires_grad=trainable)
        self.w_and  = nn.Parameter(torch.tensor([ 2.0,  2.0]), requires_grad=trainable)
        self.b_and  = nn.Parameter(torch.tensor(-1.5     ), requires_grad=trainable)
        self.w_nor  = nn.Parameter(torch.tensor([-2.0, -2.0]), requires_grad=trainable)
        self.b_nor  = nn.Parameter(torch.tensor( 1.0     ), requires_grad=trainable)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        l_nand = torch.sigmoid(x @ self.w_nand + self.b_nand)
        l_and  = torch.sigmoid(x @ self.w_and  + self.b_and)
        l_nor  = torch.sigmoid(x @ self.w_nor  + self.b_nor)
        return torch.stack([l_nand, l_and, l_nor], dim=1)  # (batch,3)


# ─── 4. PolicyNet ──────────────────────────────────────────────────────
class PolicyNet(nn.Module):
    def __init__(self, beta: float = 10.0):
        super().__init__()
        self.beta = beta

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return F.softmax(logits * self.beta, dim=1)  # (batch,3)


# ─── 5. TradingPolicy ──────────────────────────────────────────────────
class TradingPolicy(nn.Module):
    def __init__(self,
                 window_short: int,
                 window_long:  int,
                 beta:         float = 10.0,
                 input_trainable:   bool  = True,   # now always trainable
                 feature_trainable: bool  = True):  # now always trainable
        super().__init__()
        # 1) Build submodules
        self.input_f = InputFeature(window_short,
                                    window_long,
                                    trainable=input_trainable)
        self.feat_n  = FeatureNet(trainable=feature_trainable)
        # 2) Logic gates are *always* frozen
        self.logic   = LogicNet(trainable=False)
        self.policy  = PolicyNet(beta)

        # 3) Random‐init only the trainable parts
        for name, param in self.named_parameters():
            if param.requires_grad and not name.startswith("logic"):
                nn.init.normal_(param, mean=0.0, std=0.05)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.squeeze(-1)
        f1 = self.input_f(x)
        f2 = self.feat_n(f1)
        f3 = self.logic(f2)
        return self.policy(f3)




