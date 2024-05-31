import torch
import torch.nn as nn


class SimpleNet(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(1, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 1),
        )

    def forward(self, x):
        return self.seq(x)




class ExpansionLayer(nn.Module):
    def __init__(self, din, dout, k):
        super().__init__()
        self.fc1 = nn.Linear(din, dout * k)
        self.fc2 = nn.Linear(dout * k, dout)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x


class ExpansionMLP(nn.Module):
    def __init__(self, d, k):
        super().__init__()
        self.seq = nn.Sequential(
            ExpansionLayer(1, d, k),
            ExpansionLayer(d, d, k),
            ExpansionLayer(d, d, k),
            ExpansionLayer(d, 1, k),
        )

    def forward(self, x):
        return self.seq(x)


class LearnedActivation(nn.Module):
    def __init__(self, size, k):
        super().__init__()
        self.k = k
        self.w1 = nn.Parameter(torch.Tensor(size, k))
        self.b1 = nn.Parameter(torch.Tensor(size, k))
        self.w2 = nn.Parameter(torch.Tensor(size, k))
        self.b2 = nn.Parameter(torch.Tensor(size))
        self.reset_parameters()

    def reset_parameters(self):
        s = 1 / self.k**.5
        nn.init.uniform_(self.w1, a=-1, b=1)
        nn.init.uniform_(self.w2, a=-s, b=s)
        nn.init.zeros_(self.b1)
        nn.init.zeros_(self.b2)

    def forward(self, x):
        x = torch.einsum("bd,dk->bdk", x, self.w1) + self.b1
        return torch.einsum("bdk,dk->bd", torch.relu(x), self.w2) + self.b2


class LearnedActivationMLP(nn.Module):
    def __init__(self, d=100, k=3):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(1, d),
            LearnedActivation(d, k),
            nn.Linear(d, d),
            LearnedActivation(d, k),
            nn.Linear(d, d),
            LearnedActivation(d, k),
            nn.Linear(d, 1),
        )

    def forward(self, x):
        return self.seq(x)


class KanLayer(nn.Module):
    def __init__(self, din, dout, k):
        super().__init__()
        self.din = din
        self.dout = dout
        self.w1 = nn.Parameter(torch.Tensor(din, dout, k))
        self.b1 = nn.Parameter(torch.Tensor(din, dout, k))
        self.w2 = nn.Parameter(torch.Tensor(din, dout, k))
        self.b2 = nn.Parameter(torch.Tensor(dout))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w1)
        nn.init.xavier_uniform_(self.w2)
        nn.init.zeros_(self.b1)
        nn.init.zeros_(self.b2)

    def forward(self, x):
        x = torch.einsum("bi,iok->biok", x, self.w1) + self.b1
        x = torch.einsum("biok,iok->bo", torch.relu(x), self.w2) + self.b2
        return x


class Kan(nn.Module):
    def __init__(self, d=100, k=3):
        super().__init__()
        self.seq = nn.Sequential(
            KanLayer(1, d, k),
            KanLayer(d, d, k),
            KanLayer(d, d, k),
            nn.Linear(d, 1),
        )

    def forward(self, x):
        return self.seq(x)


class RegluLayer(nn.Module):
    def __init__(self, din, dout, func=torch.relu):
        super().__init__()
        self.func = func
        self.fc1 = nn.Linear(din, dout)
        self.fc2 = nn.Linear(din, dout)

    def forward(self, x):
        return self.func(self.fc1(x)) * self.fc2(x)


class RegluMLP(nn.Module):
    def __init__(self, d=100, func=torch.relu):
        super().__init__()
        self.seq = nn.Sequential(
            RegluLayer(1, d, func),
            RegluLayer(d, d, func),
            RegluLayer(d, d, func),
            nn.Linear(d, 1, func),
        )

    def forward(self, x):
        return self.seq(x)


class RegluExpandMLP(nn.Module):
    def __init__(self, d=100, k=3):
        super().__init__()
        self.seq = nn.Sequential(
            RegluLayer(1, k * d),
            nn.Linear(k * d, d),
            nn.ReLU(),
            RegluLayer(d, k * d),
            nn.Linear(k * d, d),
            nn.ReLU(),
            RegluLayer(d, k * d),
            nn.Linear(k * d, d),
            nn.ReLU(),
            nn.Linear(d, 1),
        )

    def forward(self, x):
        return self.seq(x)


class Mix2Layer(nn.Module):
    def __init__(self, din, dout, k, func=torch.relu):
        super().__init__()
        self.func = func
        self.k = k
        self.w1 = nn.Parameter(torch.Tensor(din, dout, k))
        self.b1 = nn.Parameter(torch.Tensor(dout, k))
        self.w2 = nn.Parameter(torch.Tensor(din, dout, k))
        self.b2 = nn.Parameter(torch.Tensor(dout))
        self.reset_parameters()

    def reset_parameters(self):
        d = (self.w1.size(0) + self.w1.size(1)) / 2
        s = 3**.5 / (d * self.k)**.25
        nn.init.uniform_(self.w1, a=-s, b=s)
        nn.init.uniform_(self.w2, a=-s, b=s)
        nn.init.zeros_(self.b1)
        nn.init.zeros_(self.b2)

    def forward(self, x):
        g = self.func(torch.einsum("bi,iok->bok", x, self.w1) + self.b1)
        x = torch.einsum("bi,iok,bok->bo", x, self.w2, g) + self.b2
        return x


class Mix2MLP(nn.Module):
    def __init__(self, d=100, k=3, func=torch.relu):
        super().__init__()
        self.seq = nn.Sequential(
            Mix2Layer(1, d, k, func),
            Mix2Layer(d, d, k, func),
            Mix2Layer(d, d, k, func),
            nn.Linear(d, 1),
        )

    def forward(self, x):
        return self.seq(x)
