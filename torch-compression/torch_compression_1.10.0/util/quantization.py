import torch


def uniform_noise(input):
    """U(-0.5, 0.5)"""
    return torch.empty_like(input).uniform_(-0.5, 0.5)


class Quantize(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, mode="round", mean=None):
        ctx.use_mean = False
        if mode == "noise":
            return input + uniform_noise(input)
        else:
            if mean is not None:
                input = input - mean
                ctx.use_mean = True
            return input.round()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, -grad_output if ctx.use_mean else None


def quantize(input, mode="round", mean=None):
    """quantize function"""
    return Quantize.apply(input, mode, mean)


def scale_quant(input, scale=2**8):
    return quantize(input * scale) / scale


def noise_quant(input):
    return quantize(input, mode='noise')


def gumbel_like(input, eps: float = 1e-20):  # ~Gumbel(0, 1)
    exp = torch.rand_like(input).add_(eps).log_().neg_()
    return exp.add_(eps).log_().neg_()


def gumbel_softmax(logits, tau: float = 1, dim: int = -1):
    """Samples from the `Gumbel-Softmax distribution`. ~Gumbel(logits, tau)"""
    gumbels = (logits + gumbel_like(logits)) / tau
    return gumbels.softmax(dim)


class StochasticGumbelAnnealing(torch.nn.Module):

    def __init__(self, init_tau=0.5, c=0.001, iter=0, eps=1e-5):
        super().__init__()
        self.register_buffer("tau", torch.FloatTensor([init_tau]))
        self.register_buffer("iter", torch.IntTensor([iter]))
        self.c = c
        self.eps = eps

        self.step()

    def extra_repr(self):
        return f"tau={self.tau.item():.3e}, c={self.c:.2e}, iter={self.iter.item()}"

    def forward(self, input):
        bound = torch.stack([input.floor(), input.ceil()], dim=-1).detach_()
        distance = bound.sub(input.unsqueeze(-1)).abs()
        logits = torch.atan(distance.clamp_max(1-self.eps))/(-self.tau)
        weight = gumbel_softmax(logits, self.tau, dim=-1)
        output = (bound * weight).sum(dim=-1)
        return output

    def get_tau(self):
        decay = self.c * (self.iter-700)
        return 0.5*torch.exp(-decay.clamp_min(0)).item()

    def step(self, closure=None):
        value = self.get_tau() if closure is None else closure(self.iter)
        self.tau.data.fill_(value).clamp_min_(1e-12)
        self.iter.data += 1


def random_quant(input, m=noise_quant, mean=None, p=0.5):
    """use `m` method random quantize input with  probability `p`, others use round"""
    idxs = torch.rand_like(input).lt(p).bool()

    output = torch.empty_like(input)
    output.masked_scatter_(idxs, m(input.masked_select(idxs)))

    round_idx = torch.logical_not(idxs)
    if mean is not None:
        mean = mean.masked_select(round_idx)
        output.masked_scatter_(round_idx, quantize(input.masked_select(
            round_idx), mode='round', mean=mean) + mean)
    else:
        output.masked_scatter_(round_idx, quantize(input.masked_select(
            round_idx), mode='round'))

    return output
