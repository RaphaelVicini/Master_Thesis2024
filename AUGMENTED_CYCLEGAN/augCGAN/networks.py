import functools

import numpy as np
import torch
import torch.nn as nn

from modules import ResnetBlock, CondInstanceNorm, TwoInputSequential, CINResnetBlock, InstanceNorm2d


###############################################################################
# Functions
###############################################################################

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(InstanceNorm2d, affine=True)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def define_G(input_nc, output_nc, ngf, n_blocks, norm='instance', which_model_netG='resnet',
             use_dropout=False, gpu_ids=[]):
    netG = None
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())

    norm_layer = get_norm_layer(norm_type=norm)

    netG = ResnetGenerator(input_nc, output_nc, ngf, n_blocks, norm_layer=norm_layer,
                           use_dropout=use_dropout, gpu_ids=gpu_ids)

    if len(gpu_ids) > 0:
        netG.cuda()
    netG.apply(weights_init)
    return netG


def define_stochastic_G(nlatent, input_nc, output_nc, ngf, n_blocks, norm='instance',
                        which_model_netG='resnet', use_dropout=False, gpu_ids=[]):
    netG = None
    use_gpu = len(gpu_ids) > 0

    if use_gpu:
        assert (torch.cuda.is_available())

    norm_layer = CondInstanceNorm

    netG = CINResnetGenerator(nlatent, input_nc, output_nc, ngf, n_blocks, norm_layer=norm_layer,
                              use_dropout=use_dropout, gpu_ids=gpu_ids)

    if len(gpu_ids) > 0:
        netG.cuda()
    netG.apply(weights_init)
    return netG


def define_D_A(input_nc, ndf, which_model_netD, norm, use_sigmoid=False, num_D=2, gpu_ids=[]):
    netD = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert (torch.cuda.is_available())

    netD = MultiscaleDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid, num_D=num_D, gpu_ids=gpu_ids)

    if use_gpu:
        netD.cuda()
    netD.apply(weights_init)
    return netD


def define_D_B(input_nc, ndf, which_model_netD, norm, use_sigmoid=False, num_D=2, gpu_ids=[]):
    netD = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert (torch.cuda.is_available())

    netD = MultiscaleDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid,  num_D=num_D, gpu_ids=gpu_ids)

    if use_gpu:
        netD.cuda()
    netD.apply(weights_init)
    return netD


def define_LAT_D(nlatent, ndf, use_sigmoid=False, gpu_ids=[]):
    netD = None
    use_gpu = len(gpu_ids) > 0

    if use_gpu:
        assert (torch.cuda.is_available())

    netD = DiscriminatorLatent(nlatent, ndf, gpu_ids=gpu_ids)

    if use_gpu:
        netD.cuda()
    netD.apply(weights_init)
    return netD


def define_E(nlatent, input_nc, loadSize, nef, norm='batch', gpu_ids=[]):
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert (torch.cuda.is_available())
    netE = LatentEncoder(nlatent, input_nc, loadSize, nef, norm_layer=norm_layer, gpu_ids=gpu_ids)

    if use_gpu:
        netE.cuda()
    netE.apply(weights_init)
    return netE


def print_network(net, out_f=None):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    # print(net)
    if out_f is not None:
        out_f.write(net.__repr__() + "\n")
        out_f.write('Total number of parameters: %d\n' % num_params)
        out_f.flush()
    return num_params


##############################################################################
# Network Classes
##############################################################################

######################################################################
# Modified version of ResnetGenerator that supports stochastic mappings
# using Conditonal instance norm (can support CBN easily)
######################################################################
class CINResnetGenerator(nn.Module):
    def __init__(self, nlatent, input_nc, output_nc, ngf=64, n_blocks=9, norm_layer=CondInstanceNorm,
                 use_dropout=False, gpu_ids=[], padding_type='reflect'):
        assert (n_blocks >= 0)
        super(CINResnetGenerator, self).__init__()
        self.gpu_ids = gpu_ids

        instance_norm = functools.partial(InstanceNorm2d, affine=True)

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, stride=1, bias=True),
            norm_layer(ngf, nlatent),
            nn.ReLU(True),

            nn.Conv2d(ngf, 2 * ngf, kernel_size=3, padding=1, stride=1, bias=True),
            norm_layer(2 * ngf, nlatent),
            nn.ReLU(True),

            nn.Conv2d(2 * ngf, 4 * ngf, kernel_size=3, padding=1, stride=2, bias=True),
            norm_layer(4 * ngf, nlatent),
            nn.ReLU(True)
        ]

        for i in range(n_blocks):
            model += [CINResnetBlock(x_dim=4 * ngf, z_dim=nlatent, padding_type=padding_type,
                                     norm_layer=norm_layer, use_dropout=use_dropout, use_bias=True)]

        model += [
            nn.ConvTranspose2d(4 * ngf, 2 * ngf, kernel_size=3, stride=2, padding=1,
                               output_padding=1, bias=True),
            norm_layer(2 * ngf, nlatent),
            nn.ReLU(True),

            nn.Conv2d(2 * ngf, ngf, kernel_size=3, padding=1, stride=1, bias=True),
            norm_layer(ngf, nlatent),
            nn.ReLU(True),

            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=3),
            nn.Tanh()
        ]

        self.model = TwoInputSequential(*model)

    def forward(self, input, noise):
        if len(self.gpu_ids) > 1 and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, (input, noise), self.gpu_ids)
        else:
            return self.model(input, noise)


######################################################################
# ResnetGenerator for deterministic mappings
######################################################################
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=9, norm_layer=InstanceNorm2d, use_dropout=False,
                 gpu_ids=[], padding_type='reflect'):
        assert (n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.gpu_ids = gpu_ids

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, stride=1, bias=True),
            norm_layer(ngf),
            nn.ReLU(True),

            nn.Conv2d(ngf, 2 * ngf, kernel_size=3, padding=1, stride=1, bias=True),
            norm_layer(2 * ngf),
            nn.ReLU(True),

            nn.Conv2d(2 * ngf, 4 * ngf, kernel_size=3, padding=1, stride=2, bias=True),
            norm_layer(4 * ngf),
            nn.ReLU(True),
        ]

        for i in range(n_blocks):
            model += [ResnetBlock(4 * ngf, padding_type=padding_type, norm_layer=norm_layer,
                                  use_dropout=use_dropout, use_bias=True)]

        model += [

            nn.ConvTranspose2d(4 * ngf, 2 * ngf,
                               kernel_size=3, stride=2,
                               padding=1, output_padding=1,
                               bias=True),
            norm_layer(2 * ngf),
            nn.ReLU(True),

            nn.Conv2d(2 * ngf, ngf, kernel_size=3, padding=1, bias=True),
            norm_layer(ngf),
            nn.ReLU(True),

            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=3),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if len(self.gpu_ids) > 1 and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


######################################################################
# Discriminator that supports stochastic mappings
# using Conditonal instance norm (can support CBN easily)
######################################################################
class CINDiscriminator(nn.Module):
    def __init__(self, nlatent, input_nc, ndf=64, norm_layer=nn.BatchNorm2d,
                 use_sigmoid=False, gpu_ids=[]):
        """
        nlatent: number of channles in both latent codes (or one of them - depending on the model)
        input_nc: number of channels in input and output (assumes both inputs are concatenated)
        """
        super(CINDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids

        use_bias = True

        kw = 4
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(ndf, 2 * ndf, kernel_size=kw, stride=2, padding=1, bias=use_bias),
            norm_layer(2 * ndf, nlatent),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(2 * ndf, 4 * ndf,
                      kernel_size=kw, stride=1, padding=1, bias=use_bias),
            norm_layer(4 * ndf, nlatent),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(4 * ndf, 5 * ndf,
                      kernel_size=kw, stride=1, padding=1, bias=use_bias),
            norm_layer(5 * ndf, nlatent),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(5 * ndf, 1, kernel_size=kw, stride=1, padding=1)
        ]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = TwoInputSequential(*sequence)

    def forward(self, input, noise):
        if len(self.gpu_ids) > 1 and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, (input, noise), self.gpu_ids)
        else:
            return self.model(input, noise)


######################################################################
# Discriminator network
######################################################################

class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d,
                 use_sigmoid=False, num_D=2, gpu_ids=[]):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers

        for i in range(num_D):
            netD = Discriminator(input_nc, ndf, norm_layer, gpu_ids)
            setattr(self, 'layer' + str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        return [model(input)]

    def forward(self, input):
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            model = getattr(self, 'layer' + str(num_D - 1 - i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D - 1):
                input_downsampled = self.downsample(input_downsampled)
        return result


class Discriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, gpu_ids=[]):
        """
        nlatent: number of channles in both latent codes (or one of them - depending on the model)
        input_nc: number of channels in input and output (assumes both inputs are concatenated)
        """
        super(Discriminator, self).__init__()
        self.gpu_ids = gpu_ids

        use_bias = True

        kw = 4
        padw = 1
        n_layers = 3
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        if len(self.gpu_ids) > 1 and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


class Discriminator_edges(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d,
                 use_sigmoid=False, gpu_ids=[]):
        """
        nlatent: number of channles in both latent codes (or one of them - depending on the model)
        input_nc: number of channels in input and output (assumes both inputs are concatenated)
        """
        super(Discriminator_edges, self).__init__()
        self.gpu_ids = gpu_ids

        use_bias = True

        kw = 3
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(ndf, 2 * ndf, kernel_size=kw, stride=2, padding=1, bias=use_bias),
            norm_layer(2 * ndf),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(2 * ndf, 4 * ndf, kernel_size=kw, stride=2, padding=1, bias=use_bias),
            norm_layer(4 * ndf),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(4 * ndf, 4 * ndf, kernel_size=kw, stride=2, padding=1, bias=use_bias),
            norm_layer(4 * ndf),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(4 * ndf, 1, kernel_size=4, stride=1, padding=0, bias=True)
        ]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        if len(self.gpu_ids) > 1 and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


class DiscriminatorLatent(nn.Module):
    def __init__(self, nlatent, ndf,
                 use_sigmoid=False, gpu_ids=[]):
        super(DiscriminatorLatent, self).__init__()

        self.gpu_ids = gpu_ids
        self.nlatent = nlatent

        use_bias = True
        sequence = [
            nn.Linear(nlatent, ndf),
            nn.BatchNorm1d(ndf),
            nn.LeakyReLU(0.2, True),

            nn.Linear(ndf, ndf),
            nn.BatchNorm1d(ndf),
            nn.LeakyReLU(0.2, True),

            nn.Linear(ndf, ndf),
            nn.BatchNorm1d(ndf),
            nn.LeakyReLU(0.2, True),

            nn.Linear(ndf, 1)
        ]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        if input.dim() == 4:
            input = input.view(input.size(0), self.nlatent)

        if len(self.gpu_ids) > 1 and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


######################################################################
# Encoder network for latent variables
######################################################################
class LatentEncoder(nn.Module):
    def __init__(self, nlatent, input_nc, loadSize, nef, norm_layer, gpu_ids=[]):
        super(LatentEncoder, self).__init__()
        self.gpu_ids = gpu_ids
        use_bias = False

        kw = 3
        n_downsampling = int(np.floor(np.log(loadSize) / np.log(2))) - 3
        sequence = [nn.Conv2d(input_nc, nef, kernel_size=kw, stride=2, padding=1, bias=True),
                    nn.ReLU(True)]

        for i in range(n_downsampling):
            mult = 2 ** i
            sequence += [nn.Conv2d(nef * mult, 2 * nef * mult, kernel_size=kw, stride=2, padding=1, bias=use_bias),
                         norm_layer(2 * nef * mult),
                         nn.ReLU(True)]

        sequence += [nn.Conv2d(nef * 2 ** n_downsampling, nef * 2 ** n_downsampling, kernel_size=4, stride=1, padding=0,
                               bias=use_bias),
                     norm_layer(nef * 2 ** n_downsampling),
                     nn.ReLU(True)]

        self.conv_modules = nn.Sequential(*sequence)

        # make sure we return mu and logvar for latent code normal distribution
        self.enc_mu = nn.Conv2d(nef * 2 ** n_downsampling, nlatent, kernel_size=1, stride=1, padding=0, bias=True)
        self.enc_logvar = nn.Conv2d(nef * 2 ** n_downsampling, nlatent, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, input):
        if len(self.gpu_ids) > 1 and isinstance(input.data, torch.cuda.FloatTensor):
            conv_out = nn.parallel.data_parallel(self.conv_modules, input, self.gpu_ids)
            mu = nn.parallel.data_parallel(self.enc_mu, conv_out, self.gpu_ids)
            logvar = nn.parallel.data_parallel(self.enc_logvar, conv_out, self.gpu_ids)
        else:
            conv_out = self.conv_modules(input)
            mu = self.enc_mu(conv_out)
            logvar = self.enc_logvar(conv_out)
        return (mu.view(mu.size(0), -1), logvar.view(logvar.size(0), -1))
