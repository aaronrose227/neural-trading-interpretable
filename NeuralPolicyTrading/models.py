import torch
import torch.nn as nn
import torch.nn.functional as F

class InputFeature(nn.Module):
    def __init__(self, window_short: int, window_long: int, trainable: bool):
        super().__init__()
        self.window_short = window_short
        self.window_long  = window_long

        # 1) EWMA over last window_short points
        self.lin_short = nn.Linear(window_short, 1, bias=True)
        # 2) SMA over window_long points
        self.lin_long  = nn.Linear(window_long,  1, bias=True)

        # initialize EWMA weights: α = 2/(w+1)
        α = 2.0 / (window_short + 1.0)
        # build vector [α*(1-α)^(w-1), ..., α*(1-α)^1, α*(1-α)^0]
        pow_terms = torch.arange(window_short-1, -1, -1, dtype=torch.float32)
        w_short = α * (1 - α) ** pow_terms
        with torch.no_grad():
            self.lin_short.weight.copy_(w_short.unsqueeze(0))
            self.lin_short.bias.zero_()

        # initialize SMA weights: uniform 1/window_long
        with torch.no_grad():
            self.lin_long.weight.fill_(1.0 / window_long)
            self.lin_long.bias.zero_()

        # freeze or unfreeze exactly as requested
        for p in self.parameters():
            p.requires_grad = trainable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, window_long)
        m_tilde = self.lin_short(x[:, -self.window_short :])  # EWMA
        m       = self.lin_long(x)                            # SMA
        return torch.cat([m_tilde, m], dim=1)  # (batch,2)

class FeatureNet(nn.Module):
    def __init__(self, trainable: bool):
        super().__init__()
        # two scalar thresholds τ1, τ2 initialized to zero
        self.tau1 = nn.Parameter(torch.zeros(()))
        self.tau2 = nn.Parameter(torch.zeros(()))
        for p in (self.tau1, self.tau2):
            p.requires_grad = trainable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch,2) = [m_tilde, m]
        d = x[:, :1] - x[:, 1:]
        o1 = torch.sigmoid(d + self.tau1)
        o2 = torch.sigmoid(-d + self.tau2)
        return torch.cat([o1, o2], dim=1)  # (batch,2)

class LogicNet(nn.Module):
    def __init__(self, trainable: bool = False):
        super().__init__()
        # exact Boolean‐approximate gates per Eq.(8–10)
        # NAND: σ(-x1 - x2 + 1.5)
        self.w_nand = nn.Parameter(torch.tensor([[-1.0, -1.0]]), requires_grad=trainable)
        self.b_nand = nn.Parameter(torch.tensor([ 1.5   ]), requires_grad=trainable)
        # AND:  σ( x1 + x2 - 1.5)
        self.w_and  = nn.Parameter(torch.tensor([[ 1.0,  1.0]]), requires_grad=trainable)
        self.b_and  = nn.Parameter(torch.tensor([-1.5   ]), requires_grad=trainable)
        # NOR:  σ(-x1 - x2 + 0.5)
        self.w_nor  = nn.Parameter(torch.tensor([[-1.0, -1.0]]), requires_grad=trainable)
        self.b_nor  = nn.Parameter(torch.tensor([ 0.5   ]), requires_grad=trainable)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch,2)
        n = torch.sigmoid(x @ self.w_nand.T + self.b_nand)
        a = torch.sigmoid(x @ self.w_and .T + self.b_and)
        r = torch.sigmoid(x @ self.w_nor .T + self.b_nor)
        return torch.cat([n, a, r], dim=1)  # (batch,3)

class PolicyNet(nn.Module):
    def __init__(self, beta: float = 10.0):
        super().__init__()
        self.beta = beta

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        # soft‐argmax with temperature β
        return F.softmax(logits * self.beta, dim=1)

class TradingPolicy(nn.Module):
    def __init__(self,
                 window_short: int,
                 window_long:  int,
                 beta:         float = 10.0,
                 input_trainable:   bool  = False,
                 feature_trainable: bool  = False,
                 logic_trainable:   bool  = False):
        super().__init__()
        self.input_f = InputFeature(window_short, window_long, trainable=input_trainable)
        self.feat_n  = FeatureNet(trainable=feature_trainable)
        self.logic   = LogicNet(trainable=logic_trainable)
        self.policy  = PolicyNet(beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # accept both (batch,window,1) or (batch,window)
        if x.ndim == 3:
            x = x.squeeze(-1)
        f1 = self.input_f(x)
        f2 = self.feat_n(f1)
        f3 = self.logic(f2)
        return self.policy(f3)
