import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

def save_slice_as_png(nifti_file, slice_index, axis, output_png):
    # Charger le fichier NIfTI
    img = nib.load(nifti_file)
    data = img.get_fdata()

    # Sélectionner la slice selon l'axe spécifié
    if axis == 'x':
        slice_data = data[slice_index, :, :]
    elif axis == 'y':
        slice_data = data[:, slice_index, :]
    elif axis == 'z':
        slice_data = data[:, :, slice_index]
    else:
        raise ValueError("L'axe doit être 'x', 'y' ou 'z'.")

    # Normaliser les données pour qu'elles soient dans l'intervalle [0, 1]
    slice_data = (slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data))

    # Enregistrer la slice comme image PNG
    plt.imsave(output_png, slice_data, cmap='gray')

# Exemple d'utilisation
input=r"C:\Users\rapha\OneDrive\Bureau\Mémoire\augmented_cyclegan-private-version\augmented_cyclegan-mr2ct-master\results_latest\['400761867']_latest.nii"
output= r"C:\Users\rapha\OneDrive\Bureau\Mémoire\augmented_cyclegan-private-version\augmented_cyclegan-mr2ct-master\results_latest\['400761867']_latest.png"


save_slice_as_png(input, 50, 'z', output)
#400709048
#400761867
#401411138