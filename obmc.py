import torch

from util.sampler import shift, warp
import util.functional as FE


class OPMC(torch.nn.Module):

    aux_channel = {0: 0, 1: 12}

    def __init__(self, obmc_type):
        super().__init__()
        self.obmc_type = obmc_type

    @staticmethod
    def make_obmc(input, Flow):
        obmc_volume = [warp(input, Flow)]
        for bias in [[0, 1], [1, 0], [0, -1], [-1, 0]]:
            motion = torch.Tensor(bias).view(1, 2).expand(input.size(0), -1)
            _flow = shift(Flow, motion)
            obmc_volume.append(warp(input, _flow))

        return torch.cat(obmc_volume, dim=1)

    def forward(self, input, Flow):
        if self.obmc_type == 1:
            return self.make_obmc(input, Flow)
        elif self.obmc_type == 2:
            block_size = []
            for d in range(2, input.dim()):
                block_size.append(input.size(d)//Flow.size(d))

            blocks = FE.space_to_depth(input, block_size)
            warped = warp(blocks, Flow)
            return FE.depth_to_space(warped, block_size)
