import nibabel as nib
import numpy as np

# def crop_center_3d(image, cropx, cropy):
#     x, y, z = image.shape
#     startx = x // 2 - (cropx // 2)
#     starty = y // 2 - (cropy // 2)
#     return image[startx:startx + cropx, starty:starty + cropy, :]

def calculate_mae_me(image1, image2):
    mae = np.mean(np.abs(image1 - image2))
    me = np.mean(image1 - image2)
    return mae, me

def main(image1_path, image2_path, output_path):
    # Load the NIfTI images
    img1 = nib.load(image1_path)
    img2 = nib.load(image2_path)

    # Get the image data as numpy arrays
    img1_data = img1.get_fdata()
    img2_data = img2.get_fdata()

    # Ensure the images are 3D
    assert img1_data.ndim == 3, "Image 1 is not 3D"
    assert img2_data.ndim == 3, "Image 2 is not 3D"


    # assert img2_data.shape[0:2] == (256, 256), "Image 2 is not 256x256 in the x and y dimensions"
    # assert img1_data.shape[0:2] == (256, 256), "Cropped Image 1 is not 256x256 in the x and y dimensions"

    img2_data_aligned = img2_data[:, :, :]

    # Calculate the MAE and ME
    mae, me = calculate_mae_me(img1_data, img2_data_aligned)

    # Print the MAE and ME
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Error (ME): {me}")

    # Subtract the images
    result_data = img1_data - img2_data_aligned

    # Create new NIfTI images
    result_img = nib.Nifti1Image(result_data, img1.affine)

    # Save the new NIfTI images
    nib.save(result_img, output_path)

# Usage example
main(r"C:\Users\rapha\OneDrive\Bureau\wetransferUGATIT\results_maison_real\['401411138']_transformed.nii\['401411138']_transformed.nii",
     r"C:\Users\rapha\OneDrive\Bureau\Mémoire\augmented_cyclegan-private-version\augmented_cyclegan-mr2ct-master\results\['401411138']_resi_cropped.nii",
     r"C:\Users\rapha\OneDrive\Bureau\Mémoire\augmented_cyclegan-private-version\augmented_cyclegan-mr2ct-master\results\['401411138']_Mean_augm_cropped.nii")
#400709048
#400761867
#401411138