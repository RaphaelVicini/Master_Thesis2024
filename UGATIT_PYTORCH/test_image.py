# Copyright 2020 Lorna Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import argparse
import os
import random
import time
import numpy as np
import cv2

import torch.backends.cudnn as cudnn
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image
from datalaoderTEST import CustomPatientDatasetTESTING
from torch.utils.data import DataLoader
import torchio as tio
from ugatit_pytorch import Generator

parser = argparse.ArgumentParser(description="PyTorch Generate Realistic Animation Face.")
parser.add_argument("--file", type=str, default="assets/testA_1.jpg",
                    help="Selfie image name. (default:`assets/testA_1.jpg`)")
parser.add_argument("--model-name", type=str, default="ct2mr-26-04-cropped")
parser.add_argument("--cuda", action="store_true",default=True, help="Enables cuda")
parser.add_argument("--image-size", type=int, default=256,
                    help="size of the data crop (squared assumed). (default:256)")
parser.add_argument("--manualSeed", type=int,
                    help="Seed for initializing training. (default:none)")

args = parser.parse_args()
print(args)

if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
print("Random Seed: ", args.manualSeed)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda:0" if args.cuda else "cpu")

def adjust_and_resize(fake_A_slice):

    denorm_slice = ((fake_A_slice + 1) * (1000 + 150) / 2) - 150
    # Convertir en tensor si ce n'est pas déjà le cas
    denorm_tensor = denorm_slice.clone().detach()
    # Si vous souhaitez également travailler sur CPU et assurez-vous que le tensor est en flottant
    denorm_tensor = denorm_tensor.to(dtype=torch.float32, device='cpu')
    # Créer une image Scalar pour utiliser avec TorchIO pour le redimensionnement
    scalar_image = tio.ScalarImage(tensor=denorm_tensor)  # Ajoute les dimensions batch et channel
    # Utiliser TorchIO pour redimensionner
    resize_transform = tio.Resize([1,256,256])
    resized_image = resize_transform(scalar_image)
    return resized_image.data


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

def cam(x, size = 256):
    x = x - np.min(x)
    cam_img = x / np.max(x)
    cam_img = np.uint8(255 * cam_img)
    cam_img = cv2.resize(cam_img, (size, size))
    cam_img = cv2.applyColorMap(cam_img, cv2.COLORMAP_JET)
    return cam_img / 255.0

# create model
model = Generator(image_size=args.image_size).to(device)

# Load state dicts
model.load_state_dict(torch.load(r"C:\Users\rapha\OneDrive\Bureau\wetransferUGATIT\model\netG_B2A.pth"))

# Set model mode
model.eval()



test_dataset = CustomPatientDatasetTESTING(root_dir=r"C:\Users\rapha\OneDrive\Bureau\wetransferUGATIT\DATA")
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
for i, data in enumerate(test_dataloader):
    real_A_3D, real_B_3D = data['ct'][tio.DATA], data['irm'][tio.DATA]  # real_A_3D est toute l'images en 3D
    print(f"traitement des patients: {data['name']}")
    transformed_slices = []

    #for d in range(real_A_3D.size(-1)):
    real_image_A = real_A_3D[:, :, :, :, 50]  # chaque slice de real_A_3D
    real_image_B = real_B_3D[:, :, :, :, 50]
    #print(d)

    real_image_A, real_image_B = real_image_A.to(device), real_image_B.to(device)
    # GRAY to RGB
    # real_image_A = real_image_A.repeat(1, 3, 1, 1)
    # real_image_B = real_image_B.repeat(1, 3, 1, 1)


    with torch.no_grad():
        real_image_B_cropped = cropped256(real_image_B)
        real_image_A_cropped = cropped256(real_image_A)
        fake_A_slice , HM, _  = model(real_image_B_cropped)

        #resized_fake_A_slice = adjust_and_resize(fake_A_slice)


    # Convertir le tenseur en objet PIL Image


    # Enregistrer l'image en tant que fichier PNG

    #     transformed_slices.append(fake_A_slice.cpu().squeeze(0))
    #
    # volume_transformed_3d = torch.stack(transformed_slices, dim=-1)
    # fake_CT = tio.ScalarImage(tensor=volume_transformed_3d)
    #
    output_directory = os.path.join('heatmaps')
    output_filepath = os.path.join(output_directory, f"{data['name']}_HM_last.png")

    os.makedirs(output_directory, exist_ok=True)
    cam_output = cam(HM.cpu().numpy())
    cam_tensor = torch.tensor(cam_output).permute(2, 0, 1)
    vutils.save_image(cam_tensor, output_filepath)

    #
    # fake_CT.save(output_filepath)

