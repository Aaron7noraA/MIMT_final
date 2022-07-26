import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from .vision import plot_yuv, YUV4202RGB
_debug = False
_check_grad = False


def torchseed(seed=666):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def check_shape(*tensors):
    for t in tensors:
        if torch.is_tensor(t):
            print(t.shape)
        elif isinstance(t, (list, tuple)):
            check_shape(*t)


def debug(*args, **kwargs):
    if _debug:
        print(*args, **kwargs)


@torch.no_grad()
def check_range(input, name=""):
    msg = name+" %.4f, %.4f, %.4f, %.4f, %.4f" % \
        (input.min(), input.median(), input.mean(), input.var(), input.max())
    if _debug:
        print(msg)
    return msg


def max_norm(input):
    flattened = input.flatten(2)
    shifted = flattened.sub(flattened.mean(-1, keepdim=True))
    return shifted.div(shifted.abs().max(-1, keepdim=True)[0].add(1e-9)).view_as(input)


grad_mem = {}


class CheckGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, name: str, input, plot: bool = False, figdir: str = './'):
        ctx.name = name
        return input

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output.device not in grad_mem:
            grad_mem[grad_output.device] = []
        grad_mem[grad_output.device].append(
            (ctx.name, grad_output.mean().item(), grad_output.abs().max().item()))
        return None, grad_output, None, None


def check_grad(name: str, input, plot: bool = False, figdir: str = './'):
    if _check_grad:
        return CheckGrad.apply(name, input, plot, figdir)
    else:
        return input


def dump_grad(path=None):
    grad = {}
    for item in grad_mem.values():
        for key, mean, max_ in item[::-1]:
            if key not in grad:
                grad[key] = (mean/len(grad_mem), max_)
            else:
                grad[key] = (grad[key][0]+mean/len(grad_mem),
                             max(grad[key][1], max_))
    grad_mem.clear()
    if path is not None:
        with open(path, "w") as fp:
            for key, (mean, max_) in grad.items():
                fp.write(key+": mean={:.5e}, max={:.5e}\n".format(mean, max_))
    return grad


class GlobalVisualizer():

    def __init__(self):
        self.mean = None
        self.var = None
        self.queue = []

    @torch.no_grad()
    def set_mean_var(self, input):
        self.mean = input.flatten(1).mean(1).view(-1, 1, 1, 1)
        self.var = input.flatten(1).var(1).view(-1, 1, 1, 1)

    @torch.no_grad()
    def normed_visual(self, input, figname):
        save_image(F.layer_norm(input, input.size()[
                   1:])*self.var + self.mean, figname)

    @torch.no_grad()
    def queue_visual(self, input, figname):
        # save_image(input, figname)
        if input.size(1) == 6: # YUV420 w/ space_to_depth(Y)
            input = YUV4202RGB(input)
        self.queue.append(input)


    def plot_queue(self, figname, nrow=3):
        save_image(torch.cat(self.queue), figname, nrow=nrow, padding=5)
        del self.queue
        self.queue = []

    @torch.no_grad()
    def visual(self, input, figname):
        save_image(input, figname)


visualizer = GlobalVisualizer()


class GlobalVisualizerYUV():

    def __init__(self):
        self.mean = None
        self.var = None
        self.queue = []

    @torch.no_grad()
    def set_mean_var(self, input):
        self.mean = input.flatten(1).mean(1).view(-1, 1, 1, 1)
        self.var = input.flatten(1).var(1).view(-1, 1, 1, 1)

    @torch.no_grad()
    def normed_visual(self, input, figname):
        save_image(F.layer_norm(input, input.size()[
                   1:])*self.var + self.mean, figname)

    @torch.no_grad()
    def queue_visual(self, input, figname):
        # save_image(input, figname)
        if input.size(1) == 6: # YUV420 w/ space_to_depth(Y)
            input = plot_yuv(input)
        self.queue.append(input)


    def plot_queue(self, figname, nrow=3):
        save_image(torch.cat(self.queue), figname, nrow=nrow, padding=5)
        del self.queue
        self.queue = []

    @torch.no_grad()
    def visual(self, input, figname):
        save_image(input, figname)

visualizer_yuv = GlobalVisualizerYUV()


class GlobalLogger():
    def __init__(self):
        self.fp = None

    def open(self, fp, mode='w'):
        """open"""
        if isinstance(fp, str):
            fp = open(fp, mode)
        if self.fp is not None:
            raise RuntimeError()
        else:
            self.fp = fp

    def write(self, s="", end="\n"):
        """write"""
        self.fp.write(s+end)

    def close(self):
        self.fp.flush()
        self.fp.close()
        self.fp = None


logger = GlobalLogger()
