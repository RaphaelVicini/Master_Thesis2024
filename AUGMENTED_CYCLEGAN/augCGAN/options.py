import argparse
import pickle as pkl
import os

import torch


def create_sub_dirs(opt, sub_dirs):
    for sub_dir in sub_dirs:
        dir_path = os.path.join(opt.expr_dir, sub_dir)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        setattr(opt, sub_dir, dir_path)


class TrainOptions(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--dataroot', type=str, required=False, help='path to data')
        self.parser.add_argument('--name', type=str, required=True,
                                 help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints/', help='models are saved here')

        # data
        self.parser.add_argument('--loadSize', type=int, default=256, help='image size')
        self.parser.add_argument('--crop', action='store_true', help='crop image')
        self.parser.add_argument('--cropSize', type=int, default=128, help='then crop to this size')
        self.parser.add_argument('--input_nc', type=int, default=1, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=1, help='# of output image channels')
        self.parser.add_argument('--numpy_data', type=int, choices=[0, 1], default=0,
                                 help='use numpy data instead of default PNG images')
        self.parser.add_argument('--nThreads', default=1, type=int, help='# threads for loading data')

        # exp
        self.parser.add_argument('--seed', type=int, help='manual seed')
        self.parser.add_argument('--model', type=str, choices=['aug_cycle_gan', 'cycle_gan'], default='aug_cycle_gan',
                                 help='which model to train')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

        # supervised training
        self.parser.add_argument('--supervised', action='store_true',
                                 help='fraction of training data for supervised training')
        self.parser.add_argument('--sup_frac', type=float, default=0.1,
                                 help='fraction of training data for supervised training')
        self.parser.add_argument('--lambda_sup_A', type=float, default=0.1, help='weight for supervised loss (B -> A)')
        self.parser.add_argument('--lambda_sup_B', type=float, default=0.1, help='weight for supervised loss (A -> B)')

        self.parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        self.parser.add_argument('--continue_train', action='store_true',
                                 help='continue training: load the latest model')
        self.parser.add_argument('--load_pretrain', type=str, default='',
                                 help='load the pretrained model from the specified location')
        self.parser.add_argument('--which_epoch', type=str, default='latest',
                                 help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--epoch_count', type=int, default=1,
                                 help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        self.parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=100,
                                 help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')

        # model
        self.parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        self.parser.add_argument('--nef', type=int, default=64, help='# of encoder filters in first conv layer')
        self.parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        self.parser.add_argument('--n_blocks', type=int, default=9,
                                 help='number of residual blocks in the global generator network')
        self.parser.add_argument('--nlatent', type=int, default=32,
                                 help='# of latent code dimensions. Used only for stochastic models, e.g. cycle_ali')
        self.parser.add_argument('--which_model_netD', type=str, default='basic', help='selects model to use for netD')
        self.parser.add_argument('--num_D', type=int, default=2, help='# of discrim')
        self.parser.add_argument('--which_model_netG', type=str, default='resnet', help='selects model to use for netG')
        self.parser.add_argument('--norm', type=str, default='instance',
                                 help='instance normalization or batch normalization')
        self.parser.add_argument('--use_dropout', action='store_true', help='use dropout for the generator')
        self.parser.add_argument('--max_gnorm', type=float, default=500.,
                                 help='max grad norm to which it will be clipped (if exceeded)')
        self.parser.add_argument('--stoch_enc', action='store_true', help='use a stochastic encoder')
        self.parser.add_argument('--z_gan', type=int, default=1, choices=[0, 1], help='use a GAN on z_B')
        self.parser.add_argument('--enc_A_B', type=int, default=1, choices=[0, 1],
                                 help='encoder of z_B conditoned on both A and B')

        self.parser.add_argument('--no_lsgan', action='store_true',
                                 help='do *not* use least square GAN, if false, use vanilla GAN')
        self.parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
        self.parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
        self.parser.add_argument('--lambda_z_B', type=float, default=0.025, help='weight for cycle loss (B -> A -> B)')

        # monitoring
        self.parser.add_argument('--monitor_gnorm', type=bool, default=True, help='flag set to monitor grad norms')
        self.parser.add_argument('--display_freq', type=int, default=5000,
                                 help='frequency of showing training results on screen')
        self.parser.add_argument('--print_freq', type=int, default=100,
                                 help='frequency of showing training results on console')
        self.parser.add_argument('--save_epoch_freq', type=int, default=5,
                                 help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--num_multi', type=int, default=1,
                                 help='the number of z_B used to generate different B')
        self.parser.add_argument('--eval_A_freq', type=int, default=1, help='frequency of evaluating on A')
        self.parser.add_argument('--eval_B_freq', type=int, default=1, help='frequency of evaluating on B')

        self.parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        self.parser.add_argument('--display_server', type=str, default="http://localhost",
                                 help='visdom server of the web display')
        self.parser.add_argument('--display_env', type=str, default='main',
                                 help='visdom display environment name (default is "main")')
        self.parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        self.parser.add_argument('--update_html_freq', type=int, default=1000,
                                 help='frequency of saving training results to html')

        self.initialized = True

    def parse(self, sub_dirs=None):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        if self.opt.continue_train:
            name = self.opt.name
            dataroot = self.opt.dataroot
            opt_path = os.path.join(self.opt.load_pretrain, 'opt.pkl')
            model_path = os.path.join(self.opt.load_pretrain, 'latest')
            self.opt.__dict__.update(parse_opt_file(opt_path))
            self.opt.continue_train = True
            self.opt.load_pretrain = model_path
            self.opt.name = name
            self.opt.dataroot = dataroot
        else:
            str_ids = self.opt.gpu_ids.split(',')
            self.opt.gpu_ids = []
            for str_id in str_ids:
                id = int(str_id)
                if id >= 0:
                    self.opt.gpu_ids.append(id)

            # Set gpu ids
            if len(self.opt.gpu_ids) > 0:
                torch.cuda.set_device(self.opt.gpu_ids[0])

        # save to the disk
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        self.opt.expr_dir = expr_dir

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')
        if not os.path.exists(expr_dir):
            os.makedirs(expr_dir)

        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')

        file_name = os.path.join(expr_dir, 'opt.pkl')
        with open(file_name, 'w') as opt_file:
            pkl.dump(args, opt_file)

        # create sub dirs
        if sub_dirs is not None:
            create_sub_dirs(self.opt, sub_dirs)

        return self.opt


class TestOptions(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--chk_path', required=False, type=str,
                                 help='path to checkpoint -- we assume expr_dir is containing dir', default=r"C:\Users\rapha\OneDrive\Bureau\Mémoire\models_augmented\latest")
        self.parser.add_argument('--res_dir', type=str, default='test_res',
                                 help='results directory (will create under expr_dir)')
        self.parser.add_argument('--train_logvar', type=int, default=1, help='train logvar_B on training data')
        self.parser.add_argument('--dataroot', required=False, type=str, help='path to images')
        self.parser.add_argument('--metric', default='visual', type=str, choices=['bpp', 'mse', 'visual', 'noise_sens'])

        # data
        self.parser.add_argument('--loadSize', type=int, default=256, help='image size')
        self.parser.add_argument('--crop', action='store_true', help='crop image')
        self.parser.add_argument('--cropSize', type=int, default=128, help='then crop to this size')
        self.parser.add_argument('--input_nc', type=int, default=1, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=1, help='# of output image channels')
        self.parser.add_argument('--numpy_data', type=int, choices=[0, 1], default=0,
                                 help='use numpy data instead of default PNG images')
        self.parser.add_argument('--nThreads', default=1, type=int, help='# threads for loading data')

        # exp
        self.parser.add_argument('--seed', type=int, help='manual seed')
        self.parser.add_argument('--model', type=str, choices=['aug_cycle_gan', 'cycle_gan'], default='aug_cycle_gan',
                                 help='which model to train')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

        # supervised training
        self.parser.add_argument('--supervised', action='store_true',
                                 help='fraction of training data for supervised training')
        self.parser.add_argument('--sup_frac', type=float, default=0.1,
                                 help='fraction of training data for supervised training')
        self.parser.add_argument('--lambda_sup_A', type=float, default=0.1, help='weight for supervised loss (B -> A)')
        self.parser.add_argument('--lambda_sup_B', type=float, default=0.1, help='weight for supervised loss (A -> B)')

        self.parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        self.parser.add_argument('--continue_train', action='store_true',
                                 help='continue training: load the latest model')
        self.parser.add_argument('--load_pretrain', type=str, default='',
                                 help='load the pretrained model from the specified location')
        self.parser.add_argument('--which_epoch', type=str, default='latest',
                                 help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--epoch_count', type=int, default=1,
                                 help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        self.parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=100,
                                 help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')

        # model
        self.parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        self.parser.add_argument('--nef', type=int, default=64, help='# of encoder filters in first conv layer')
        self.parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        self.parser.add_argument('--n_blocks', type=int, default=9,
                                 help='number of residual blocks in the global generator network')
        self.parser.add_argument('--nlatent', type=int, default=32,
                                 help='# of latent code dimensions. Used only for stochastic models, e.g. cycle_ali')
        self.parser.add_argument('--which_model_netD', type=str, default='basic', help='selects model to use for netD')
        self.parser.add_argument('--num_D', type=int, default=2, help='# of discrim')
        self.parser.add_argument('--which_model_netG', type=str, default='resnet', help='selects model to use for netG')
        self.parser.add_argument('--norm', type=str, default='instance',
                                 help='instance normalization or batch normalization')
        self.parser.add_argument('--use_dropout', action='store_true', help='use dropout for the generator')
        self.parser.add_argument('--max_gnorm', type=float, default=500.,
                                 help='max grad norm to which it will be clipped (if exceeded)')
        self.parser.add_argument('--stoch_enc', action='store_true', help='use a stochastic encoder')
        self.parser.add_argument('--z_gan', type=int, default=1, choices=[0, 1], help='use a GAN on z_B')
        self.parser.add_argument('--enc_A_B', type=int, default=1, choices=[0, 1],
                                 help='encoder of z_B conditoned on both A and B')

        self.parser.add_argument('--no_lsgan', action='store_true',
                                 help='do *not* use least square GAN, if false, use vanilla GAN')
        self.parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
        self.parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
        self.parser.add_argument('--lambda_z_B', type=float, default=0.025, help='weight for cycle loss (B -> A -> B)')

        # monitoring
        self.parser.add_argument('--monitor_gnorm', type=bool, default=True, help='flag set to monitor grad norms')
        self.parser.add_argument('--display_freq', type=int, default=5000,
                                 help='frequency of showing training results on screen')
        self.parser.add_argument('--print_freq', type=int, default=100,
                                 help='frequency of showing training results on console')
        self.parser.add_argument('--save_epoch_freq', type=int, default=5,
                                 help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--num_multi', type=int, default=1,
                                 help='the number of z_B used to generate different B')
        self.parser.add_argument('--eval_A_freq', type=int, default=1, help='frequency of evaluating on A')
        self.parser.add_argument('--eval_B_freq', type=int, default=1, help='frequency of evaluating on B')

        self.parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        self.parser.add_argument('--display_server', type=str, default="http://localhost",
                                 help='visdom server of the web display')
        self.parser.add_argument('--display_env', type=str, default='main',
                                 help='visdom display environment name (default is "main")')
        self.parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        self.parser.add_argument('--update_html_freq', type=int, default=1000,
                                 help='frequency of saving training results to html')


    def parse(self):
        return self.parser.parse_args()


def parse_opt_file(opt_path):
    def parse_val(s):
        if s == 'None':
            return None
        if s == 'True':
            return True
        if s == 'False':
            return False
        if s == 'inf':
            return float('inf')
        try:
            f = float(s)
            # special case
            if '.' in s:
                return f
            i = int(f)
            return i if i == f else f
        except ValueError:
            return s

    opt = None
    with open(opt_path) as f:
        if opt_path.endswith('pkl'):
            opt = pkl.load(f)
        else:
            opt = dict()
            for line in f:
                if line.startswith('-----'):
                    continue
                k, v = line.split(':')
                opt[k.strip()] = parse_val(v.strip())
    return opt
