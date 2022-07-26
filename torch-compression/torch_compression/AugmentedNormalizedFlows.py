import torch
from torch import nn

from torch_compression.util.math import lower_bound
from torch_compression.util.quantization import quantize
from torch_compression.util.toolbox import *
from torch_compression.util.vision import fft_visual

plot_flow = PlotFlow().cuda()

class AugmentedNormalizedFlow(nn.Sequential):

    def __init__(self, *args, use_code, transpose, distribution='gaussian', clamp=1, integerlize=False):
        super(AugmentedNormalizedFlow, self).__init__(*args)
        self.use_code = use_code
        self.transpose = transpose
        self.distribution = distribution
        if distribution == 'gaussian':
            self.init_code = torch.randn_like
        elif distribution == 'uniform':
            self.init_code = torch.rand_like
        elif distribution == 'zeros':
            self.init_code = torch.zeros_like
        self.clamp = clamp
        self.train_scale = False
        self.integerlize = integerlize


    def extra_repr(self):
        return "clamp={clamp}".format(**self.__dict__)

    def get_condition(self, input, jac=None, layer=-1):
        prefix = self.name+("" if layer < 0 or str.isdigit(self.name[-1]) else "_"+str(layer)
                            )+"_"
        condition = super().forward(check_grad(prefix+'model_input', input))
        if self.use_code:
            loc, scale = condition.chunk(2, dim=1)
        else:
            loc, scale = condition, input.new_zeros(condition.shape)
        if self.train_scale and self.use_code:
            debug("TC")
            loc = loc.detach()

        self.scale = lower_bound(scale, 0.11)
        debug('C')
        self.jacobian(jac, prefix=prefix)

        loc = check_grad(prefix+'loc', loc)
        scale = check_grad(prefix+'scale', self.scale)
        condition = torch.cat([loc, scale], dim=1)
        return condition, jac

    def forward(self, input, code=None, jac=None, rev=False, last_layer=False, layer=-1, visual=False, figname='', shift_img=True):
        if visual:
            debug(figname)
        if self.transpose:
            debug("T1")
            input, code = code, input
        prefix = self.name+("" if layer < 0 or str.isdigit(self.name[-1]) else "_"+str(layer)
                            )+"_"+("rev_" if rev else "")
        condition = super().forward(check_grad(prefix+'model_input', input))
        if self.use_code:
            loc, scale = condition.chunk(2, dim=1)
        else:
            loc, scale = condition, input.new_zeros(input.size(0), 1)
        if self.train_scale and self.use_code:
            debug("TC")
            loc = loc.detach()
        if self.integerlize:
            debug("Q")
            loc = quantize(loc)

        self.scale = self.clamp * (scale.sigmoid() * 2 - 1)
        loc = check_grad(prefix+'loc', loc)
        scale = check_grad(prefix+'scale', self.scale)

        if code is None:
            if self.use_code:
                debug('I')
                code = self.init_code(loc)
            else:
                debug('I0')
                code = None

        # if code is not None:
        #     check_range(code, 'code')
        # check_range(loc, 'loc')
        # check_range(scale.exp(), 'scale')

        if visual:
            logger.write(check_range(loc, prefix+"loc"))
            logger.write(check_range(scale, prefix+"logscale"))
            logger.write(check_range(scale.exp(), prefix+"scale"))
            logger.write()

        debug('LL', last_layer)
        if code is not None:
            code = check_grad(prefix+'code', code)

        if (not rev) ^ self.transpose:
            debug('F')
            if code is None:
                debug('SK')

                code = loc

                if visual and code.size(1) == 2:
                    debug('visual')
                    visualizer.queue_visual(
                        plot_flow(loc).div(255.)+0.5 if self.integerlize else plot_flow(loc), figname+'_loc.png')
                    # fft_visual(loc, figname+'_loc.png')
                elif visual and code.size(1) == 3:
                    debug('visual')
                    visualizer.queue_visual(
                        loc.div(255.)+0.5 if self.integerlize else loc+0.5*shift_img, figname+'_loc.png')
                    # fft_visual(loc, figname+'_loc.png')
                elif visual and code.size(1) == 6:
                    debug('visual')
                    visualizer.queue_visual(
                        loc.div(255.)+0.5 if self.integerlize else loc+0.5*shift_img, figname+'_loc.png')
                    visualizer_yuv.queue_visual(
                        loc.div(255.)+0.5 if self.integerlize else loc+0.5*shift_img, figname+'_loc.png')
                    # fft_visual(loc, figname+'_loc.png')

            else:

                if self.use_code and not last_layer:
                    code = code * scale.exp()
                    self.jacobian(jac, rev=rev, prefix=prefix)

                if visual and code.size(1) == 2:
                    debug('visual')
                    visualizer.queue_visual(
                        plot_flow(code).div(255.) if self.integerlize else plot_flow(code), figname+'.png')
                    # fft_visual(loc, figname+'_loc.png')
                elif visual and code.size(1) == 3:
                    debug('visual')
                    visualizer.queue_visual(
                        code.div(255.) if self.integerlize else code, figname+'.png')
                    # fft_visual(code, figname+'.png')
                elif visual and code.size(1) == 6:
                    debug('visual')
                    visualizer.queue_visual(
                        code.div(255.) if self.integerlize else code, figname+'.png')
                    visualizer_yuv.queue_visual(
                        code.div(255.) if self.integerlize else code, figname+'.png')
                    # fft_visual(loc, figname+'_loc.png')

                code = code + loc

                
                if visual and code.size(1) == 2:
                    debug('visual')
                    visualizer.queue_visual(
                        plot_flow(loc).div(255.)+0.5 if self.integerlize else plot_flow(loc), figname+'_loc.png')
                    visualizer.queue_visual(
                        plot_flow(code).div(255.) if self.integerlize else plot_flow(code), figname+'_out.png')
                elif visual and code.size(1) == 3:
                    debug('v_out')
                    visualizer.queue_visual(
                        # loc.div(255.)+(0 if figname[-1] == '0' else 0.5) if self.integerlize else loc+(0 if figname[-1] == '0' else 0.5), 
                        #        figname+'_loc.png')
                        loc.div(255.)+0.5 if self.integerlize else loc+0.5*shift_img, figname+'_loc.png')
                    visualizer.queue_visual(
                        code.div(255.) if self.integerlize else code, figname+'_out.png')
                    # fft_visual(loc, figname+'_loc.png')
                    # fft_visual(code, figname+'_out.png')

                    # visualizer.normed_visual(code, figname+'_out_norm.png')
                    # visualizer.normed_visual(loc, figname+'_loc_norm.png')
                elif visual and code.size(1) == 6:
                    debug('v_out')
                    visualizer.queue_visual(
                        loc.div(255.)+0.5 if self.integerlize else loc+0.5*shift_img, figname+'_loc.png')
                    visualizer.queue_visual(
                        code.div(255.) if self.integerlize else code, figname+'_out.png')
                    visualizer_yuv.queue_visual(
                        loc.div(255.)+0.5 if self.integerlize else loc+0.5*shift_img, figname+'_loc.png')
                    visualizer_yuv.queue_visual(
                        code.div(255.) if self.integerlize else code, figname+'_out.png')
                    
        else:
            debug('rF')

            code = code - loc

            if visual and code.size(1) == 2:
                debug('visual')
                visualizer.queue_visual(
                    plot_flow(loc).div(255.)+0.5 if self.integerlize else plot_flow(loc), figname+'_loc.png')
                visualizer.queue_visual(
                    plot_flow(code).div(255.) if self.integerlize else plot_flow(code), figname+'.png')
            elif visual and code.size(1) == 3:
                debug('visual')
                visualizer.queue_visual(
                    loc.div(255.)+0.5 if self.integerlize else loc+0.5*shift_img, figname+'_loc.png')
                visualizer.queue_visual(
                    code.div(255.) if self.integerlize else code, figname+'.png')
            elif visual and code.size(1) == 6:
                debug('visual')
                visualizer.queue_visual(
                    loc.div(255.)+0.5 if self.integerlize else loc+0.5*shift_img, figname+'_loc.png')
                visualizer.queue_visual(
                    code.div(255.) if self.integerlize else code, figname+'.png')
                visualizer_yuv.queue_visual(
                    loc.div(255.)+0.5 if self.integerlize else loc+0.5*shift_img, figname+'_loc.png')
                visualizer_yuv.queue_visual(
                    code.div(255.) if self.integerlize else code, figname+'.png')

            if self.use_code and not last_layer:
                code = code / scale.exp()
                self.jacobian(jac, rev=rev, prefix=prefix)

                if visual and code.size(1) == 2:
                    debug('v_out')
                    visualizer.queue_visual(
                        plot_flow(code).div(255.) if self.integerlize else plot_flow(code), figname+'_out.png')
                    # fft_visual(code, figname+'_out.png')
                elif visual and code.size(1) == 3:
                    debug('v_out')
                    visualizer.queue_visual(
                        code.div(255.) if self.integerlize else code, figname+'_out.png')
                    # fft_visual(code, figname+'_out.png')
                elif visual and code.size(1) == 6:
                    debug('v_out')
                    visualizer.queue_visual(
                        code.div(255.) if self.integerlize else code, figname+'_out.png')
                    visualizer_yuv.queue_visual(
                        code.div(255.) if self.integerlize else code, figname+'_out.png')

            else:
                debug('SK')

        # check_range(code, 'out')
        # debug("")

        if self.transpose:
            debug("T2")
            input, code = code, input
        return input, code, jac

    def jacobian(self, jacs=None, rev=False, prefix=""):
        if jacs is not None:
            jac = check_grad(prefix+'scale_jac', self.scale).flatten(1).sum(1)
            if rev ^ self.transpose:
                debug('JR')
                jac = jac * -1
            else:
                debug("J")

            jacs.append(jac)
        else:
            jac = None
        return jac


class CondAugmentedNormalizedFlow(nn.Module):
    def __init__(self, use_code, transpose, distribution='gaussian', clamp=1, integerlize=False):
        super(CondAugmentedNormalizedFlow, self).__init__()
        self.use_code = use_code
        self.transpose = transpose
        self.distribution = distribution
        if distribution == 'gaussian':
            self.init_code = torch.randn_like
        elif distribution == 'uniform':
            self.init_code = torch.rand_like
        elif distribution == 'zeros':
            self.init_code = torch.zeros_like
        self.clamp = clamp
        self.train_scale = False
        self.integerlize = integerlize

    def extra_repr(self):
        return "clamp={clamp}".format(**self.__dict__)

    def get_condition(self, input, input_cond, jac=None, layer=-1):
        prefix = self.name+("" if layer < 0 or str.isdigit(self.name[-1]) else "_"+str(layer)
                            )+"_"

        condition = self.net_forward(check_grad(prefix+'model_input', input), input_cond)

        if self.use_code:
            loc, scale = condition.chunk(2, dim=1)
        else:
            loc, scale = condition, input.new_zeros(input.size(0), 1)
        if self.train_scale and self.use_code:
            debug("TC")
            loc = loc.detach()

        self.scale = lower_bound(scale, 0.11)
        debug('C')
        self.jacobian(jac, prefix=prefix)

        loc = check_grad(prefix+'loc', loc)
        scale = check_grad(prefix+'scale', self.scale)
        condition = torch.cat([loc, scale], dim=1)
        return condition, jac

    def net_forward(self, input, input_cond):

        raise NotImplementedError
    
    def affine_forward(self, input, condition, code=None, jac=None, rev=False, last_layer=False, layer=-1, visual=False, figname='', shift_img=True):
        prefix = self.name+("" if layer < 0 or str.isdigit(self.name[-1]) else "_"+str(layer)
                            )+"_"+("rev_" if rev else "")
        if self.use_code:
            loc, scale = condition.chunk(2, dim=1)
        else:
            loc, scale = condition, input.new_zeros(input.size(0), 1)
        if self.train_scale and self.use_code:
            debug("TC")
            loc = loc.detach()
        if self.integerlize:
            debug("Q")
            loc = quantize(loc)

        self.scale = self.clamp * (scale.sigmoid() * 2 - 1)
        loc = check_grad(prefix+'loc', loc)
        scale = check_grad(prefix+'scale', self.scale)

        if code is None:
            if self.use_code:
                debug('I')
                code = self.init_code(loc)
            else:
                debug('I0')
                code = None

        # if code is not None:
        #     check_range(code, 'code')
        # check_range(loc, 'loc')
        # check_range(scale.exp(), 'scale')

        if visual:
            logger.write(check_range(loc, prefix+"loc"))
            logger.write(check_range(scale, prefix+"logscale"))
            logger.write(check_range(scale.exp(), prefix+"scale"))
            logger.write()

        debug('LL', last_layer)
        if code is not None:
            code = check_grad(prefix+'code', code)

        if (not rev) ^ self.transpose:
            debug('F')
            if code is None:
                debug('SK')

                code = loc

                if visual and code.size(1) == 3:
                    debug('visual')
                    visualizer.queue_visual(
                        loc.div(255.)+0.5 if self.integerlize else loc+0.5*shift_img, figname+'_loc.png')
                    # fft_visual(loc, figname+'_loc.png')
                elif visual and code.size(1) == 6:
                    debug('visual')
                    visualizer.queue_visual(
                        loc.div(255.)+0.5 if self.integerlize else loc+0.5*shift_img, figname+'_loc.png')
                    visualizer_yuv.queue_visual(
                        loc.div(255.)+0.5 if self.integerlize else loc+0.5*shift_img, figname+'_loc.png')
                    # fft_visual(loc, figname+'_loc.png')

            else:

                if self.use_code and not last_layer:
                    code = code * scale.exp()
                    self.jacobian(jac, rev=rev, prefix=prefix)

                if visual and code.size(1) == 3:
                    debug('visual')
                    visualizer.queue_visual(
                        code.div(255.) if self.integerlize else code, figname+'.png')
                    # fft_visual(code, figname+'.png')
                elif visual and code.size(1) == 6:
                    debug('visual')
                    visualizer.queue_visual(
                        code.div(255.) if self.integerlize else code, figname+'.png')
                    visualizer_yuv.queue_visual(
                        code.div(255.) if self.integerlize else code, figname+'.png')
                    # fft_visual(loc, figname+'_loc.png')

                code = code + loc

                if visual and code.size(1) == 3:
                    debug('v_out')
                    visualizer.queue_visual(
                        #loc.div(255.)+(0 if figname[-1] == '0' else 0.5) if self.integerlize else loc+(0 if figname[-1] == '0' else 0.5), figname+'_loc.png')
                        loc.div(255.)+0.5 if self.integerlize else loc+0.5*shift_img, figname+'_loc.png')
                    visualizer.queue_visual(
                        code.div(255.) if self.integerlize else code, figname+'_out.png')
                    # fft_visual(loc, figname+'_loc.png')
                    # fft_visual(code, figname+'_out.png')

                    # visualizer.normed_visual(code, figname+'_out_norm.png')
                    # visualizer.normed_visual(loc, figname+'_loc_norm.png')
                elif visual and code.size(1) == 6:
                    debug('v_out')
                    visualizer.queue_visual(
                        loc.div(255.)+0.5 if self.integerlize else loc+0.5*shift_img, figname+'_loc.png')
                    visualizer.queue_visual(
                        code.div(255.) if self.integerlize else code, figname+'_out.png')
                    visualizer_yuv.queue_visual(
                        loc.div(255.)+0.5 if self.integerlize else loc+0.5*shift_img, figname+'_loc.png')
                    visualizer_yuv.queue_visual(
                        code.div(255.) if self.integerlize else code, figname+'_out.png')
                    
        else:
            debug('rF')

            code = code - loc

            if visual and code.size(1) == 3:
                debug('visual')
                visualizer.queue_visual(
                    loc.div(255.)+0.5 if self.integerlize else loc+0.5*shift_img, figname+'_loc.png')
                visualizer.queue_visual(
                    code.div(255.) if self.integerlize else code, figname+'.png')
            elif visual and code.size(1) == 6:
                debug('visual')
                visualizer.queue_visual(
                    loc.div(255.)+0.5 if self.integerlize else loc+0.5*shift_img, figname+'_loc.png')
                visualizer.queue_visual(
                    code.div(255.) if self.integerlize else code, figname+'.png')
                visualizer_yuv.queue_visual(
                    loc.div(255.)+0.5 if self.integerlize else loc+0.5*shift_img, figname+'_loc.png')
                visualizer_yuv.queue_visual(
                    code.div(255.) if self.integerlize else code, figname+'.png')

            if self.use_code and not last_layer:
                code = code / scale.exp()
                self.jacobian(jac, rev=rev, prefix=prefix)

                if visual and code.size(1) == 3:
                    debug('v_out')
                    visualizer.queue_visual(
                        code.div(255.) if self.integerlize else code, figname+'_out.png')
                    # fft_visual(code, figname+'_out.png')
                elif visual and code.size(1) == 6:
                    debug('v_out')
                    visualizer.queue_visual(
                        code.div(255.) if self.integerlize else code, figname+'_out.png')
                    visualizer_yuv.queue_visual(
                        code.div(255.) if self.integerlize else code, figname+'_out.png')

            else:
                debug('SK')

        # check_range(code, 'out')
        # debug("")

        if self.transpose:
            debug("T2")
            input, code = code, input

        return input, code, jac

    def forward(self, input, input_cond, code=None, jac=None, rev=False, last_layer=False, layer=-1, visual=False, figname='', shift_img=True):
        prefix = self.name+("" if layer < 0 or str.isdigit(self.name[-1]) else "_"+str(layer)
                            )+"_"+("rev_" if rev else "")
        if visual:
            debug(figname)
        if self.transpose:
            debug("T1")
            input, code = code, input
        
        condition = self.net_forward(check_grad(prefix+'model_input', input), input_cond)
        
        input, code, jac = self.affine_forward(input, condition, code, jac, rev, last_layer, layer, visual, figname, shift_img)

        return input, code, jac

    def jacobian(self, jacs=None, rev=False, prefix=""):
        if jacs is not None:
            jac = check_grad(prefix+'scale_jac', self.scale).flatten(1).sum(1)
            if rev ^ self.transpose:
                debug('JR')
                jac = jac * -1
            else:
                debug("J")

            jacs.append(jac)
        else:
            jac = None
        return jac

