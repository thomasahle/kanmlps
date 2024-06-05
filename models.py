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
    def __init__(self, din, dout, k, scale=1):
        super().__init__()
        self.k = k
        self.scale = scale
        self.fc1 = nn.Linear(din, dout * k)
        self.fc2 = nn.Linear(dout * k, dout)
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu', mode='fan_in')
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu', mode='fan_in')

    def forward(self, x):
        x = self.fc1(x) * self.scale
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        return x


class ExpansionMLP(nn.Module):
    def __init__(self, d, k, scale=1):
        super().__init__()
        self.seq = nn.Sequential(
            ExpansionLayer(1, d, k, scale),
            ExpansionLayer(d, d, k, scale),
            ExpansionLayer(d, d, k, scale),
            nn.Linear(d, 1),
        )

    def forward(self, x):
        return self.seq(x)


class LearnedActivation(nn.Module):
    def __init__(self, size, k, func):
        super().__init__()
        self.k = k
        self.func = func
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
        # x = torch.einsum("bd,dk->bdk", x, self.w1) + self.b1
        # return torch.einsum("bdk,dk->bd", self.func(x), self.w2) + self.b2
        x = x[:, :, None] * self.w1 + self.b1
        return (self.func(x) * self.w2).sum(2) + self.b2


class LearnedActivationMLP(nn.Module):
    def __init__(self, d=100, k=3, func=torch.relu):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(1, d),
            LearnedActivation(d, k, func),
            nn.Linear(d, d),
            LearnedActivation(d, k, func),
            nn.Linear(d, d),
            LearnedActivation(d, k, func),
            nn.Linear(d, 1),
        )

    def forward(self, x):
        return self.seq(x)


class KanLayer(nn.Module):
    def __init__(self, din, dout, k, scale=.5, func=torch.relu):
        super().__init__()
        self.k = k
        self.scale = scale
        self.func = func
        self.w1 = nn.Parameter(torch.Tensor(dout, din, k))
        self.b1 = nn.Parameter(torch.Tensor(dout, din, k))
        self.w2 = nn.Parameter(torch.Tensor(dout, din, k))
        self.b2 = nn.Parameter(torch.Tensor(dout))
        self.reset_parameters()

    def reset_parameters(self):
        s1 = s2 = self.scale
        nn.init.uniform_(self.w1, a=-s1, b=s1)
        nn.init.uniform_(self.w2, a=-s2, b=s2)
        nn.init.zeros_(self.b1)
        nn.init.zeros_(self.b2)

    def forward(self, x):
        d = (self.w1.size(0) + self.w1.size(1)) / 2
        k = self.k
        # x = torch.einsum("bi,oik->boik", x, self.w1) + self.b1
        # x = torch.einsum("boik,oik->bo", torch.relu(x), self.w2) / (d*k)**.5 + self.b2
        x = x[:, None, :, None] * self.w1 + self.b1
        x = (self.func(x) * self.w2).sum((2,3)) / (d*k)**.5 + self.b2
        return x


class Kan(nn.Module):
    def __init__(self, d=100, k=3, scale=1, func=torch.relu):
        super().__init__()
        self.seq = nn.Sequential(
            KanLayer(1, d, k, scale, func),
            KanLayer(d, d, k, scale, func),
            KanLayer(d, d, k, scale, func),
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
        nn.init.uniform_(self.w1, a=-1, b=1)
        nn.init.uniform_(self.w2, a=-1, b=1)
        nn.init.zeros_(self.b1)
        nn.init.zeros_(self.b2)

    def forward(self, x):
        o, k = self.b1.shape
        d = (self.w1.size(0) + self.w1.size(1)) / 2
        #g = self.func(torch.einsum("bi,iok->bok", x, self.w1) / d**.5 + self.b1)
        #x = torch.einsum("bi,iok,bok->bo", x, self.w2, g) / k**.5 + self.b2
        g = self.func(x @ self.w1.flatten(1,2)).reshape(-1, o, k) / d**.5 + self.b1
        x = ((x @ self.w2.flatten(1,2)).reshape(-1, o, k) * g).sum(2) / k**.5 + self.b2
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
