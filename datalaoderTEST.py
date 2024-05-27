import torchio as tio
from torch.utils.data import DataLoader
import os



class CustomPatientDatasetTESTING(tio.SubjectsDataset):
    def __init__(self, root_dir):

        subjects = []
        modality = 'T2'

        for patient_number in sorted(os.listdir(root_dir), key=lambda x: int(x)):
            patient_dir = os.path.join(root_dir, patient_number)
            if not os.path.isdir(patient_dir):
                continue

            irm_path = os.path.join(patient_dir, 'MRI', patient_number + '_valid_slices.nii')
            ct_path = os.path.join(patient_dir, 'CT', patient_number + '_valid_slices.nii')
            if os.path.exists(irm_path) and os.path.exists(ct_path):
                # Charger les images pour obtenir leur forme
                ct_image = tio.ScalarImage(ct_path)
                irm_image = tio.ScalarImage(irm_path)
                num_slices = ct_image.shape[3]  # La dimension z est à l'index 3

                # Transformation pipeline
                transform = tio.Compose([
                    tio.ToCanonical(),
                    tio.Resize((512, 512, 128)),  # Ajuster dynamiquement la dimension z
                    tio.Clamp(out_min=-150, out_max=1000, exclude=['irm']),
                    tio.RescaleIntensity(out_min_max=(-1, 1)),
                ])

                subject = tio.Subject(
                    ct=ct_image,
                    irm=irm_image,
                    name=patient_number,
                )
                subjects.append(subject)


        super().__init__(subjects, transform=transform)