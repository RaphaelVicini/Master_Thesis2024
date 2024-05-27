import os
import json
import numpy as np
import pydicom
import nibabel as nib




def calculate_variance_of_dicoms(root_dir, modality='T2', threshold=0.01):
    variance_data = {}
    for patient_number in sorted(os.listdir(root_dir), key=lambda x: int(x)):
        patient_dir = os.path.join(root_dir, patient_number)
        if not os.path.isdir(patient_dir):
            print(f"Skipping {patient_dir} as it is not a directory")
            continue
        irm_path = os.path.join(patient_dir, 'MRI', modality)
        if not os.path.exists(irm_path):
            print(f"No MRI directory found at {irm_path}")
            continue

        # Collect all DICOM files
        dicom_files = [os.path.join(irm_path, f) for f in os.listdir(irm_path) if f.endswith('.dcm')]

        # Create a list of tuples (instance_number, file_path)
        dicom_files_with_instance = []
        for file_path in dicom_files:
            try:
                dicom_image = pydicom.dcmread(file_path)
                instance_number = dicom_image.InstanceNumber
                dicom_files_with_instance.append((instance_number, file_path))
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

        # Sort files by instance number
        dicom_files_sorted = sorted(dicom_files_with_instance, key=lambda x: x[0])

        first_valid_slice = None
        last_valid_slice = None

        for instance_number, file_path in dicom_files_sorted:
            try:
                dicom_image = pydicom.dcmread(file_path)
                image_array = dicom_image.pixel_array.astype(np.float32)

                # Normalize the image array to [-1, 1]
                min_val = np.min(image_array)
                max_val = np.max(image_array)
                image_array_normalized = (image_array - min_val) / (max_val - min_val)

                variance = np.var(image_array_normalized)


                if variance > threshold:
                    print(
                        f"Slice {instance_number} of patient {patient_number} has a variance above threshold: {variance} min {min_val} max {max_val}")
                    last_valid_slice = instance_number
                    if first_valid_slice is None:
                        first_valid_slice = instance_number

            except Exception as e:
                print(f"Error reading {file_path}: {e}")

        if first_valid_slice is not None and last_valid_slice is not None:
            variance_data[patient_number] = {
                'first_valid_slice': first_valid_slice,
                'last_valid_slice': last_valid_slice
            }

    # Save the results to a JSON file
    json_path = os.path.join(root_dir, 'test_data.json')
    with open(json_path, 'w') as json_file:
        json.dump(variance_data, json_file, indent=4)

    print(f"Variance data saved to {json_path}")
    return variance_data


# Replace '/path/to/your/directory' with your actual data directory path
root_dir = '/home/vicini/Documents/test_images/sorted'
#calculate_variance_of_dicoms(root_dir)


def save_valid_slices_as_numpy_IRM(root_dir, modality='T2'):
    with open("/home/vicini/Documents/DATASETS/Dataset_augmented_cycle_GAN/val_data.json", 'r') as file:
        slice_data = json.load(file)

    for patient_number in sorted(os.listdir(root_dir), key=lambda x: int(x)):
        patient_dir = os.path.join(root_dir, patient_number)
        if not os.path.isdir(patient_dir):
            continue

        irm_path = os.path.join(patient_dir, 'MRI', modality)
        if not os.path.exists(irm_path):
            print(f"No MRI directory found for patient {patient_number} at {irm_path}")
            continue

        first_valid_slice = slice_data[patient_number]['first_valid_slice']
        last_valid_slice = slice_data[patient_number]['last_valid_slice']

        # Collect all DICOM files
        dicom_files = [os.path.join(irm_path, f) for f in os.listdir(irm_path) if f.endswith('.dcm')]
        dicom_files_sorted = sorted(dicom_files, key=lambda x: pydicom.dcmread(x).InstanceNumber)

        # Read the valid slices and stack them into a 3D numpy array
        slice_arrays = []
        for file_path in dicom_files_sorted[first_valid_slice +2 :last_valid_slice - 1 ]:
            dicom_image = pydicom.dcmread(file_path)
            slice_arrays.append(dicom_image.pixel_array)

        # Convert list of 2D slices into a 3D numpy array
        slices_3D_array = np.stack(slice_arrays)

        # Save the 3D array as a numpy file
        save_path=os.path.join(patient_dir, 'MRI')
        numpy_save_path = os.path.join(save_path, f"{patient_number}_valid_slices.npy")
        np.save(numpy_save_path, slices_3D_array)
        print(f"Saved valid slices for patient {patient_number} at {numpy_save_path}")

# Replace '/path/to/your/directory' with your actual data directory path
root_dir = '/home/vicini/Documents/DATASETS/Dataset_augmented_cycle_GAN/val'
#save_valid_slices_as_numpy_IRM(root_dir)


def save_valid_slices_as_numpy_CT(root_dir, modality='T2'):
    with open("/home/vicini/Documents/DATASETS/Dataset_augmented_cycle_GAN/val_data.json", 'r') as file:
        slice_data = json.load(file)

    for patient_number in sorted(os.listdir(root_dir), key=lambda x: int(x)):
        patient_dir = os.path.join(root_dir, patient_number)
        if not os.path.isdir(patient_dir):
            continue

        # Chemin vers les images CT
        ct_path = os.path.join(patient_dir, 'CT', patient_number + '_' + 'CT')
        if not os.path.exists(ct_path):
            print(f"No CT directory found for patient {patient_number} at {ct_path}")
            continue

        first_valid_slice = slice_data[patient_number]['first_valid_slice']
        last_valid_slice = slice_data[patient_number]['last_valid_slice']

        # Collect all DICOM files
        dicom_files = [os.path.join(ct_path, f) for f in os.listdir(ct_path) if f.endswith('.dcm')]
        dicom_files_sorted = sorted(dicom_files, key=lambda x: pydicom.dcmread(x).InstanceNumber)

        # Read the valid slices and stack them into a 3D numpy array
        slice_arrays = []
        for file_path in dicom_files_sorted[first_valid_slice:last_valid_slice + 1]:
            dicom_image = pydicom.dcmread(file_path)
            slice_arrays.append(dicom_image.pixel_array)

        # Convert list of 2D slices into a 3D numpy array
        slices_3D_array = np.stack(slice_arrays)

        # Save the 3D array as a numpy file
        save_path = os.path.join(patient_dir, 'CT')
        numpy_save_path = os.path.join(save_path, f"{patient_number}_valid_slices.npy")
        np.save(numpy_save_path, slices_3D_array)
        print(f"Saved valid slices for patient {patient_number} at {numpy_save_path}")


# Replace '/path/to/your/directory' with your actual data directory path
root_dir = '/home/vicini/Documents/DATASETS/Dataset_augmented_cycle_GAN/val'
#save_valid_slices_as_numpy_CT(root_dir, modality='CT')


def read_and_analyze_numpy_files(directory):
    # Parcourir tous les fichiers dans le répertoire spécifié
    for filename in os.listdir(directory):
        if filename.endswith('.npy'):
            file_path = os.path.join(directory, filename)
            try:
                # Charger le tableau Numpy
                data = np.load(file_path)

                # Afficher des statistiques de base
                print(f"Processing file: {filename}")
                print(f"Shape of the data: {data.shape}")
                print(f"Mean of the data: {np.mean(data)}")
                print(f"Variance of the data: {np.var(data)}")
                print(f"Min: {np.min(data)}")
                print(f"Max: {np.max(data)}")

                # Vous pouvez ici effectuer d'autres analyses ou transformations
                # Par exemple, appliquer une normalisation, filtrer des données, etc.

            except Exception as e:
                print(f"Failed to read or process {file_path}: {str(e)}")

# Spécifiez le chemin du dossier contenant les fichiers .npy
directory_path = '/home/vicini/Documents/DATASETS/Dataset_augmented_cycle_GAN/val/400175534/MRI'
#read_and_analyze_numpy_files(directory_path)





def save_valid_slices_as_nifti_IRM(root_dir, modality='T2'):
    with open("/home/vicini/Documents/test_images/test_data.json", 'r') as file:
        slice_data = json.load(file)

    for patient_number in sorted(os.listdir(root_dir), key=lambda x: int(x)):
        patient_dir = os.path.join(root_dir, patient_number)
        if not os.path.isdir(patient_dir):
            continue

        irm_path = os.path.join(patient_dir, 'MRI', modality)
        if not os.path.exists(irm_path):
            print(f"No MRI directory found for patient {patient_number} at {irm_path}")
            continue

        first_valid_slice = slice_data[patient_number]['first_valid_slice']
        last_valid_slice = slice_data[patient_number]['last_valid_slice']

        dicom_files = [os.path.join(irm_path, f) for f in os.listdir(irm_path) if f.endswith('.dcm')]
        dicom_files_sorted = sorted(dicom_files, key=lambda x: pydicom.dcmread(x).InstanceNumber)

        slice_arrays = []
        for file_path in dicom_files_sorted[first_valid_slice+2:last_valid_slice - 1]:
            dicom_image = pydicom.dcmread(file_path)
            if hasattr(dicom_image, 'RescaleSlope') and hasattr(dicom_image, 'RescaleIntercept'):
                pixel_array = dicom_image.pixel_array * dicom_image.RescaleSlope + dicom_image.RescaleIntercept
            else:
                pixel_array = dicom_image.pixel_array
            # Inversion de l'axe des x pour corriger l'effet miroir
            pixel_array = np.fliplr(pixel_array)
            slice_arrays.append(pixel_array)

        slices_3D_array = np.stack(slice_arrays)
        slices_3D_array = np.transpose(slices_3D_array, (1, 2, 0))
        slices_3D_array = np.rot90(slices_3D_array, k=-1, axes=(0, 1))

        nifti_img = nib.Nifti1Image(slices_3D_array, affine=np.eye(4))
        save_path = os.path.join(patient_dir, 'MRI')
        nifti_save_path = os.path.join(save_path, f"{patient_number}_valid_slices.nii")
        nib.save(nifti_img, nifti_save_path)
        print(f"Saved valid slices for patient {patient_number} as NIfTI at {nifti_save_path}")

# Replace '/path/to/your/directory' with your actual data directory path
#root_dir = '/home/vicini/Documents/test_dose'
#save_valid_slices_as_nifti_IRM(root_dir)



def save_valid_slices_as_nifti_CT(root_dir):
    with open("/home/vicini/Documents/test_images/test_data.json", 'r') as file:
        slice_data = json.load(file)

    for patient_number in sorted(os.listdir(root_dir), key=lambda x: int(x)):
        patient_dir = os.path.join(root_dir, patient_number)
        if not os.path.isdir(patient_dir):
            continue

        ct_path = os.path.join(patient_dir, 'CT', f'{patient_number}_CT')
        if not os.path.exists(ct_path):
            print(f"No CT directory found for patient {patient_number} at {ct_path}")
            continue

        first_valid_slice = slice_data[patient_number]['first_valid_slice']
        last_valid_slice = slice_data[patient_number]['last_valid_slice']

        dicom_files = [os.path.join(ct_path, f) for f in os.listdir(ct_path) if f.endswith('.dcm')]
        dicom_files_sorted = sorted(dicom_files, key=lambda x: pydicom.dcmread(x).InstanceNumber)

        slice_arrays = []
        for file_path in dicom_files_sorted[first_valid_slice +2:last_valid_slice - 1]:
            dicom_image = pydicom.dcmread(file_path)
            rescaled_image = dicom_image.pixel_array * dicom_image.RescaleSlope + dicom_image.RescaleIntercept
            slice_arrays.append(rescaled_image)

        slices_3D_array = np.stack(slice_arrays)
        slices_3D_array = np.transpose(slices_3D_array, (1, 2, 0))
        slices_3D_array = np.rot90(slices_3D_array, k=-1, axes=(0, 1))  # 90 degrees counter-clockwise
        slices_3D_array = np.flipud(slices_3D_array)  # Apply horizontal flip to correct mirror effect

        nifti_img = nib.Nifti1Image(slices_3D_array, affine=np.eye(4))
        save_path = os.path.join(patient_dir, 'CT')
        nifti_save_path = os.path.join(save_path, f"{patient_number}_valid_slices.nii")
        nib.save(nifti_img, nifti_save_path)
        print(f"Saved valid slices for patient {patient_number} as NIfTI at {nifti_save_path}")

root_dir = '/home/vicini/Documents/test_images/sorted'
#save_valid_slices_as_nifti_CT(root_dir)



def load_nifti_file(filepath):
    nifti_img = nib.load(filepath)
    data = nifti_img.get_fdata()
    return data

# Calculer et imprimer les statistiques
def analyze_image(data):
    print("Minimum value:", np.min(data))
    print("Maximum value:", np.max(data))
    print("Mean value:", np.mean(data))
    print("Variance:", np.var(data))

filepath = '/home/vicini/Documents/DATASETS/Dataset_augmented_cycle_GAN/train/400111251/CT/400111251_valid_slices.nii'
data = load_nifti_file(filepath)
#analyze_image(data)



import os
import pydicom
import numpy as np
import nibabel as nib

def save_all_dicom_slices_as_nifti(root_dir, modality='T2'):
    for patient_number in sorted(os.listdir(root_dir), key=lambda x: int(x)):
        patient_dir = os.path.join(root_dir, patient_number)
        if not os.path.isdir(patient_dir):
            continue

        irm_path = os.path.join(patient_dir, 'MRI')
        if not os.path.exists(irm_path):
            print(f"No MRI directory found for patient {patient_number} at {irm_path}")
            continue

        dicom_files = [os.path.join(irm_path, f) for f in os.listdir(irm_path) if f.endswith('.dcm')]
        dicom_files_sorted = sorted(dicom_files, key=lambda x: pydicom.dcmread(x).InstanceNumber)

        slice_arrays = []
        for file_path in dicom_files_sorted:
            dicom_image = pydicom.dcmread(file_path)
            if hasattr(dicom_image, 'RescaleSlope') and hasattr(dicom_image, 'RescaleIntercept'):
                pixel_array = dicom_image.pixel_array * dicom_image.RescaleSlope + dicom_image.RescaleIntercept
            else:
                pixel_array = dicom_image.pixel_array
            # Inversion de l'axe des x pour corriger l'effet miroir
            pixel_array = np.fliplr(pixel_array)

            slice_arrays.append(pixel_array)

        slices_3D_array = np.stack(slice_arrays)
        slices_3D_array = np.transpose(slices_3D_array, (1, 2, 0))
        slices_3D_array = np.rot90(slices_3D_array, k=-1, axes=(0, 1))

        nifti_img = nib.Nifti1Image(slices_3D_array, affine=np.eye(4))

        nifti_save_path = os.path.join(patient_dir, f"{patient_number}_full_slices.nii")
        nib.save(nifti_img, nifti_save_path)
        print(f"Saved all DICOM slices for patient {patient_number} as NIfTI at {nifti_save_path}")

# Replace '/path/to/your/directory' with your actual data directory path
root_dir = '/home/vicini/Documents/test_dose'
#save_all_dicom_slices_as_nifti(root_dir)


import nibabel as nib
import numpy as np


def adjust_nifti_values(input_nifti_path, output_nifti_path):
    # Charger l'image NIfTI originale
    nii_img = nib.load(input_nifti_path)
    data = nii_img.get_fdata()

    # Modifier les valeurs de l'image
    adjusted_data = np.where(data <= -1024, 0, data)

    # Créer une nouvelle image NIfTI avec les données ajustées
    new_nii_img = nib.Nifti1Image(adjusted_data, nii_img.affine, nii_img.header)

    # Sauvegarder la nouvelle image NIfTI
    nib.save(new_nii_img, output_nifti_path)


# Exemple d'utilisation
input_path = "/home/vicini/Documents/test_dose/401428451/401428451_full_slices.nii"
output_path = "/home/vicini/Documents/test_dose/401428451/401428451_full_slices.nii"
adjust_nifti_values(input_path, output_path)
