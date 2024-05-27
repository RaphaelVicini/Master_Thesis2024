import math
import os
import random
import time

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import pydicom
import torch
import torchio as tio
import torchvision.utils as vutils
from pydicom import dcmwrite
from pydicom.dataset import FileMetaDataset
from torch.autograd import Variable
import torch.nn.functional as F

from ct2mr_data import AlignedIterator, UnalignedIterator, CustomPatientDatasetTESTING
from torch.utils.data import DataLoader, Dataset
from evaluate import eval_mse_A, eval_ubo_B
from model import CycleGAN, AugmentedCycleGAN
from model import log_prob_gaussian, gauss_reparametrize, log_prob_laplace, kld_std_guss
from options import TestOptions, parse_opt_file







def crop_center(tensor, new_height, new_width):
    _, _, height, width = tensor.size()
    startx = width // 2 - (new_width // 2)
    starty = height // 2 - (new_height // 2)
    return tensor[:, :, starty:starty + new_height, startx:startx + new_width]

def make_histogram(opt,values, patient, title='Pixel Values', bins=100, color='skyblue'):

    plt.figure(figsize=(10, 6))
    plt.hist(values, bins=bins, color=color, alpha=0.7)
    plt.title(title)
    plt.xlabel('Valeur des Pixels')
    plt.ylabel('Nombre de Pixels')
    plt.grid(True)
    #plt.show()
    save_path = os.path.join(opt.res_dir, f"Histo_patient_{patient}.png")
    plt.savefig(save_path)


def adjust_and_resize(fake_A_slice):

    denorm_slice = ((fake_A_slice + 1) * (1000 + 150) / 2) - 150
    # Convertir en tensor si ce n'est pas déjà le cas
    denorm_tensor = denorm_slice.clone().detach()
    # Si vous souhaitez également travailler sur CPU et assurez-vous que le tensor est en flottant
    denorm_tensor = denorm_tensor.to(dtype=torch.float32, device='cpu')

    return denorm_tensor.data


def visualize_multi_cycle(opt, real_B, model, name='multi_cycle_test.png'):
    size = real_B.size()
    images = model.generate_multi_cycle(real_B, steps=4)
    images = [img.cpu().unsqueeze(1) for img in images]
    vis_image = torch.cat(images, dim=1).view(size[0]*len(images),size[1],size[2],size[3])
    save_path = os.path.join(opt.res_dir, name)
    vutils.save_image(vis_image.cpu(), save_path,
        normalize=True, range=(-1,1), nrow=len(images))

def visualize_cycle_B_multi(opt, real_B, model, name='cycle_B_multi_test.png'):
    size = real_B.size()
    multi_prior_z_B = Variable(real_B.data.new(opt.num_multi,
        opt.nlatent, 1, 1).normal_(0, 1).repeat(size[0],1,1,1), volatile=True)
    fake_A, multi_fake_B = model.generate_cycle_B_multi(real_B, multi_prior_z_B)
    multi_fake_B = multi_fake_B.data.cpu().view(
        size[0], opt.num_multi, size[1], size[2], size[3])
    vis_multi_image = torch.cat([real_B.data.cpu().unsqueeze(1), fake_A.data.cpu().unsqueeze(1),
                                 multi_fake_B], dim=1) \
        .view(size[0]*(opt.num_multi+2),size[1],size[2],size[3])
    save_path = os.path.join(opt.res_dir, name)
    vutils.save_image(vis_multi_image.cpu(), save_path,
                      normalize=True, range=(-1,1), nrow=opt.num_multi+2)

def visualize_multi(opt, real_A, model, name='multi_test.png'):
    size = real_A.size()
    # all samples in real_A share the same prior_z_B
    multi_prior_z_B = Variable(real_A.data.new(opt.num_multi,
        opt.nlatent, 1, 1).normal_(0, 1).repeat(size[0],1,1,1), volatile=True)
    multi_fake_B = model.generate_multi(real_A.detach(), multi_prior_z_B)
    multi_fake_B = multi_fake_B.data.cpu().view(
        size[0], opt.num_multi, size[1], size[2], size[3])
    vis_multi_image = torch.cat([real_A.data.cpu().unsqueeze(1), multi_fake_B], dim=1) \
        .view(size[0]*(opt.num_multi+1),size[1],size[2],size[3])
    save_path = os.path.join(opt.res_dir, name)
    vutils.save_image(vis_multi_image.cpu(), save_path,
        normalize=True, range=(-1,1), nrow=opt.num_multi+1)

def visualize_inference(opt, real_A, real_B, model, name='inf_test.png'):
    size = real_A.size()

    real_B = real_B[:opt.num_multi]
    # all samples in real_A share the same post_z_B
    multi_fake_B = model.inference_multi(real_A.detach(), real_B.detach())
    multi_fake_B = multi_fake_B.data.cpu().view(
        size[0], opt.num_multi, size[1], size[2], size[3])

    vis_multi_image = torch.cat([real_A.data.cpu().unsqueeze(1), multi_fake_B], dim=1) \
        .view(size[0]*(opt.num_multi+1),size[1],size[2],size[3])

    vis_multi_image = torch.cat([torch.ones(1, size[1], size[2], size[3]).cpu(), real_B.data.cpu(),
                                 vis_multi_image.cpu()], dim=0)

    save_path = os.path.join(opt.res_dir, name)
    vutils.save_image(vis_multi_image.cpu(), save_path,
        normalize=True, range=(-1,1), nrow=opt.num_multi+1)

def sensitivity_to_edge_noise(opt, model, data_B, use_gpu=True):
    """This is inspired from: https://arxiv.org/pdf/1712.02950.pdf"""
    res = []
    for std in [0, 0.1, 0.2, 0.5, 1, 2, 3, 5]:
        real_B = Variable(data_B, volatile=True)
        if use_gpu:
            real_B = real_B.cuda()
        rec_B = model.generate_noisy_cycle(real_B, std)
        s = torch.abs(real_B - rec_B).sum(3).sum(2).sum(1) / (64*64*3)
        res.append(s.data.cpu().numpy().tolist())
    np.save('noise_sens', res)

def train_MVGauss_B(dataset):
    b_mean = 0
    b_var = 0
    n = 0
    for i,batch in enumerate(dataset):
        real_B = Variable(batch['B'])
        real_B = real_B.cuda()
        b_mean += real_B.mean(0, keepdim=True)
        n += 1
        print (i)
    b_mean = b_mean / n
    for i,batch in enumerate(dataset):
        real_B = Variable(batch['B'])
        real_B = real_B.cuda()
        b_var += ((real_B-b_mean)**2).mean(0, keepdim=True)
        print (i)
    b_var = b_var / n
    return b_mean, b_var

def eval_bpp_MVGauss_B(dataset, mu, logvar):
    bpp = []
    for batch in dataset:
        real_B = Variable(batch['B'])
        real_B = real_B.cuda()
        dequant = Variable(torch.zeros(*real_B.size()).uniform_(0, 1./127.5).cuda())
        real_B = real_B + dequant
        nll = -log_prob_gaussian(real_B, mu, logvar)
        nll = nll.view(real_B.size(0), -1).sum(1) + (64*64*3) * math.log(127.5)
        bpp.append(nll.mean(0).data[0] / (64*64*3*math.log(2)))
    return np.mean(bpp)

def compute_bpp_MVGauss_B(dataroot):
    # exemple dataloader:

    # trainA, trainB, devA, devB, testA, testB = load_ct2mr(dataroot)
    # train_dataset = UnalignedIterator(trainA, trainB, batch_size=200)
    train_dataset = CustomPatientDatasetTESTING(root_dir='/home/vicini/Documents/test_images/sorted')
    train_data_loader = DataLoader(train_dataset, batch_size=2, shuffle=False, drop_last=True)
    print ('#training images = %d' % len(train_data_loader))

    #test_dataset = AlignedIterator(testA, testB, batch_size=200)
    test_dataset = CustomPatientDatasetTESTING(root_dir='/home/vicini/Documents/test_images/sorted')
    test_dataset_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, drop_last=True)
    print ('#test images = %d' % len(test_dataset_loader))

    mvg_mean, mvg_var = train_MVGauss_B(train_dataset)
    mvg_logvar = torch.log(mvg_var + 1e-5)
    bpp = eval_bpp_MVGauss_B(test_dataset, mvg_mean, mvg_logvar)
    print( "MVGauss BPP: %.4f" % bpp)


def train_logvar(dataset, model, epochs=1, use_gpu=True):
    logvar_B = Variable(torch.zeros(1, 3, 64, 64).fill_(math.log(0.01)).cuda(), requires_grad=True)
    iterative_opt = torch.optim.RMSprop([logvar_B], lr=1e-2)

    for eidx in range(epochs):
        for batch in dataset:
            #real_B_3D = Variable(batch['B'])
            real_B_3D = batch['irm'][tio.DATA]
            batch_size= 4

            for d in range(batch_size):
                real_B = real_B_3D[:,:,:,:,d]
            if use_gpu:
                real_B = real_B.cuda()
            size = real_B.size()
            dequant = Variable(torch.zeros(*real_B.size()).uniform_(0, 1./127.5).cuda())
            real_B = real_B + dequant
            enc_mu = Variable(torch.zeros(size[0], model.opt.nlatent).cuda())
            enc_logvar = Variable(torch.zeros(size[0], model.opt.nlatent).fill_(math.log(0.01)).cuda())
            fake_A = model.predict_A(real_B)
            if hasattr(model, 'netE_B'):
                params = model.predict_enc_params(fake_A, real_B)
                enc_mu = Variable(params[0].data)
                if len(params) == 2:
                    enc_logvar = Variable(params[1].data)
            z_B = gauss_reparametrize(enc_mu, enc_logvar)
            fake_B = model.predict_B(fake_A, z_B)
            z_B = z_B.view(size[0], model.opt.nlatent)
            log_prob = log_prob_laplace(real_B, fake_B, logvar_B)
            log_prob = log_prob.view(size[0], -1).sum(1)
            kld = kld_std_guss(enc_mu, enc_logvar)
            ubo = (-log_prob + kld) + (64*64*3) * math.log(127.5)
            ubo_val_new = ubo.mean(0).data[0]
            kld_val = kld.mean(0).data[0]
            bpp = ubo.mean(0).data[0] / (64*64*3* math.log(2.))

            print( 'UBO: %.4f, KLD: %.4f, BPP: %.4f' % (ubo_val_new, kld_val, bpp))
            loss = ubo.mean(0)
            iterative_opt.zero_grad()
            loss.backward()
            iterative_opt.step()

    return logvar_B


def compute_train_kld(train_dataset, model):
    ### DEBUGGING KLD
    train_kl = []
    for i, batch in enumerate(train_dataset):
        real_A, real_B = Variable(batch['A']), Variable(batch['B'])
        real_A = real_A.cuda()
        real_B = real_B.cuda()
        fake_A = model.predict_A(real_B)
        params = model.predict_enc_params(fake_A, real_B)
        mu = params[0]
        train_kl.append(kld_std_guss(mu, 0.0*mu).mean(0).data[0])
        if i == 100:
            break
    print ('train KL:',np.mean(train_kl))



def cropped256(image):
    _, _, height, width = image.size()
    if height < 256 or width < 256:
        raise ValueError("L'image doit avoir des dimensions au moins égales à 256x256.")

    # Calculer les indices pour recadrer le centre de l'image
    center_h, center_w = height // 2, width // 2
    half_crop_size = 256 // 2

    start_h = center_h - half_crop_size
    end_h = center_h + half_crop_size
    start_w = center_w - half_crop_size
    end_w = center_w + half_crop_size

    # Recadrer l'image
    cropped_image = image[:, :, start_h:end_h, start_w:end_w]
    return cropped_image


def test_model():
    opt = TestOptions().parse()
    dataroot = opt.dataroot

    # extract expr_dir from chk_path
    expr_dir = os.path.dirname(opt.chk_path)
    opt_path = os.path.join(expr_dir, 'opt.pkl')

    # parse saved options...
    #opt.__dict__.update(parse_opt_file(opt_path))----------------------------------------------------------------------
    opt.expr_dir = expr_dir
    opt.dataroot = dataroot

    # hack this for now
    opt.gpu_ids = [0]

    opt.seed = 12345
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)

    # create results directory (under expr_dir)
    res_path = os.path.join(opt.expr_dir, opt.res_dir)
    opt.res_dir = res_path
    if not os.path.exists(res_path):
        os.makedirs(res_path)

    use_gpu = len(opt.gpu_ids) > 0

    # trainA, trainB, devA, devB, testA, testB = load_ct2mr(opt.dataroot)
    # sub_size = int(len(trainA) * 0.2)
    # trainA = trainA[:sub_size]
    # trainB = trainB[:sub_size]
    # train_dataset = UnalignedIterator(trainA, trainB, batch_size=200)
    #train_dataset = CustomPatientDatasetTESTING(root_dir='/home/vicini/Documents/DATASETS/Dataset_augmented_cycle_GAN/train')
    #train_data_loader = DataLoader(train_dataset, batch_size=2, shuffle=False, drop_last=True)
    #print ('#training images = %d' % len(train_data_loader))
    vis_inf = False

    # test_dataset = CustomPatientDatasetTESTING(root_dir='/home/vicini/Documents/DATASETS/Dataset_augmented_cycle_GAN/val')
    # test_dataset_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, drop_last=True)
    #
    # # test_dataset = AlignedIterator(testA, testB, batch_size=200)
    # print ('#test images = %d' % len(test_dataset_loader))

    # dev_dataset = AlignedIterator(devA, devB, batch_size=200)
   # print ('#dev images = %d' % len(dev_dataset))

    vis_inf = False
    # if opt.model == 'cycle_gan':
    #     model = CycleGAN(opt, testing=True)
    # elif opt.model == 'aug_cycle_gan':
    #     model = AugmentedCycleGAN(opt, testing=True)
    #     vis_inf = True
    # else:
    #     raise NotImplementedError('Specified model is not implemented.')
    model = AugmentedCycleGAN(opt, testing=True)
    model.load(opt.chk_path)
    #model = torch.load(opt.chk_path, map_location=torch.device('cpu'))
    #model.eval()

    # debug kl
    # compute_train_kld(train_dataset, model)

    if opt.metric == 'bpp':

        if opt.train_logvar:
            print ("training logvar_B on training data...")
            logvar_B = train_logvar(train_dataset, model)
        else:
            logvar_B = None

        print ("evaluating on test set...")
        t = time.time()
        test_ubo_B, test_bpp_B, test_kld_B = eval_ubo_B(test_dataset, model, 500,
                                                        visualize=True, vis_name='test_pred_B',
                                                        vis_path=opt.res_dir,
                                                        logvar_B=logvar_B,
                                                        verbose=True,
                                                        compute_l1=True)

        print ("TEST_BPP_B: %.4f, TIME: %.4f" % (test_bpp_B, time.time()-t))

    elif opt.metric == 'mse':
        dev_mse_A = eval_mse_A(dev_dataset, model)
        test_mse_A = eval_mse_A(test_dataset, model)
        print ("DEV_MSE_A: %.4f, TEST_MSE_A: %.4f" % (dev_mse_A, test_mse_A))

    elif opt.metric == 'visual':
        opt.num_multi = 5
        n_vis = 1
        #dev_dataset = AlignedIterator(testA, testB, batch_size=n_vis)
        dev_dataset = CustomPatientDatasetTESTING(
            root_dir=r"C:\Users\rapha\OneDrive\Bureau\wetransferUGATIT\DATA",
            )
        dev_dataset_loader = DataLoader(dev_dataset, batch_size=1, shuffle=True)
        for i, vis_data in enumerate(dev_dataset_loader):

            real_A_3D = vis_data['ct'][tio.DATA]
            real_B_3D = vis_data['irm'][tio.DATA]
            patient_id= vis_data['name']
            transformed_slices = []

            mse_list = []
            print(real_B_3D.size(-1))
            print(real_A_3D.size(-1))
            for d in range(real_B_3D.size(-1)):
                real_A = real_A_3D[:,:,:,:,d]
                real_B = real_B_3D[:,:,:,:,d]
                print(d)
                #prior_z_B = real_A.data.new(n_vis, opt.nlatent, 1, 1).normal_(0, 1)

                if use_gpu:
                    real_A = real_A.cuda()
                    real_B = real_B.cuda()
                    # prior_z_B = prior_z_B.cuda()
                    with torch.no_grad():
                        #real_image_B_cropped = cropped256(real_B)
                        fake_A_slice = model.synthetize(real_B)['fake_A']


                        # Dénormalisation ici
                        resized_fake_A_slice = adjust_and_resize(fake_A_slice)


                        transformed_slices.append(resized_fake_A_slice.cpu().squeeze(0))  # Supprimer la dimension batch ajoutée

            volume_transformed_3d = torch.stack(transformed_slices, dim=-1)

            # # Création d'un objet ScalarImage pour sauvegarder le volume transformé en tant que NIfTI
            fake_CT = tio.ScalarImage(tensor=volume_transformed_3d)
            print(fake_CT.shape)

            # # Définir le chemin de sortie pour le fichier NIfTI
            output_filepath = os.path.join(r"C:\Users\rapha\OneDrive\Bureau\Mémoire\augmented_cyclegan-private-version\augmented_cyclegan-mr2ct-master\results_latest",f"{patient_id}_latest.nii")

            fake_CT.save(output_filepath)
            print(f"patient: {patient_id} have been saved at {output_filepath}")


            #break
            # visuals = model.generate_cycle(real_A, real_B, prior_z_B)
            # visuals = model.synthetize(real_B)
            # visualize_cycle(opt, real_A, visuals, name='cycle_%d.png' % i)
            # visualize_cycle(opt, real_A, visuals, name='cycle_%d' % i)

                #fAIRe comme avant un subject ct et save le subject comme un dicom?
                # fake_A_slice = visuals['fake_A'][d, :, :]
                # save_as_dicom(opt,fake_A_slice, patient_id, d)
                #nouvelle fonction pour save les data au dessus (dicom)

                #visualize_pred(opt, visuals['fake_A'], name='cycle_%d' % i)





                # exit()
                # # visualize generated B with different z_B
                # visualize_multi(opt, real_A, model, name='multi_%d.png' % i)
                #
                # visualize_cycle_B_multi(opt, real_B, model, name='cycle_B_multi_%d.png' % i)
                #
                # visualize_multi_cycle(opt, real_B, model, name='multi_cycle_%d.png' % i)
                #
                # if vis_inf:
                #     # visualize generated B with different z_B infered from real_B
                #     visualize_inference(opt, real_A, real_B, model, name='inf_%d.png' % i)
            # else:
            #     continue

    elif opt.metric == 'noise_sens':
        sensitivity_to_edge_noise(opt, model, test_dataset.next()['B'])
    else:
        raise NotImplementedError('wrong metric!')

if __name__ == "__main__":
    test_model()
