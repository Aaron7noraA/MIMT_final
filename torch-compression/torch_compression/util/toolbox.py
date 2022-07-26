import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from .vision import plot_yuv, YUV4202RGB
from .flow_utils import PlotFlow
_debug = False
_check_grad = False


plot_flow = PlotFlow().cuda()
 

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
        elif input.size(1) == 2: # When the input image is flow
            input = plot_flow(input)
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


def load_flow(path):
    """ Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    with open(path, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if magic != 202021.25:
            raise RuntimeError('Magic number incorrect. Invalid .flo file')
        else:
            W, H = np.fromfile(f, np.int32, count=2)
            data = np.fromfile(f, np.float32, count=2*W*H)
            # Reshape data into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return torch.Tensor(data).view(H, W, 2)


def save_flow(flow, filename):
    assert flow.dim() == 3
    if flow.size(-1) != 2:
        flow = flow.permute(1, 2, 0)
    if not filename.endswith('.flo'):
        filename += '.flo'
    with open(filename, 'wb') as fp:
        np.array([202021.25], np.float32).tofile(fp)
        H, W = flow.size()[:2]
        np.array([W, H], np.int32).tofile(fp)
        np.array(flow.cpu().numpy(), np.float32).tofile(fp)


def makeColorwheel():
    """
    color encoding scheme
    adapted from the color circle idea described at
    http://members.shaw.ca/quadibloc/other/colint.htm
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = torch.zeros(ncols, 3)  # r g b

    col = 0
    # RY
    colorwheel[col:col+RY, 0] = 255
    colorwheel[col:col+RY, 1] = torch.linspace(0, 255, RY)
    col += RY

    # YG
    colorwheel[col:col+YG, 0] = torch.linspace(255, 0, YG)
    colorwheel[col:col+YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = torch.linspace(0, 255, GC)
    col += GC

    # CB
    colorwheel[col:col+CB, 1] = torch.linspace(255, 0, CB)
    colorwheel[col:col+CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = torch.linspace(0, 255, BM)
    col += BM

    # MR
    colorwheel[col:col+MR, 2] = torch.linspace(255, 0, MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel.div_(255)


class PlotFlow(torch.nn.Module):
    """
    optical flow color encoding scheme
    adapted from the color circle idea described at
    http://members.shaw.ca/quadibloc/other/colint.htm

    Shape:
        - Input: :math:`(N, H, W, 2)` or (N, 2, H, W)`
        - Output: :math:(N, 3, H, W)`

    Returns:
        Flowmap
    """

    def __init__(self):
        super().__init__()
        color_map = 1 - makeColorwheel().t().view(1, 3, 1, -1)  # inverse color
        self.register_buffer('color_map', color_map)

    def forward(self, flow, scale=1):
        if flow.size(-1) != 2:
            flow = flow.permute(0, 2, 3, 1)

        known = flow.abs() <= 1e7  # UNKNOW_FLOW_THRESHOLD
        idx_known = known[..., 0] & known[..., 1]
        flow = flow * idx_known.unsqueeze(-1)

        u, v = flow.unbind(-1)
        angle = torch.atan2(-v, -u) / np.pi
        grid = torch.stack([angle, torch.zeros_like(angle)], dim=-1)
        flowmap = F.grid_sample(self.color_map.expand(grid.size(0), -1, -1, -1), grid,
                                mode='bilinear', padding_mode='border', align_corners=True)

        radius = flow.pow(2).sum(-1).add(np.finfo(float).eps).sqrt()
        maxrad = radius.flatten(1).max(1)[0].view(-1, 1, 1)
        flowmap = 1 - radius.div(maxrad / scale).unsqueeze(1) * flowmap
        return flowmap * idx_known.unsqueeze(1)
