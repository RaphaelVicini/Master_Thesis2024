import torchio as tio
from torch.utils.data import DataLoader
import os



class CustomPatientDataset(tio.SubjectsDataset):
    def __init__(self, root_dir):

        subjects = []
        modality = 'T2'
        patch_size = (256, 256, 32)

        # Transformation pipeline
        transform = tio.Compose([
            tio.ToCanonical(),
            tio.Resize((512, 512, 128)),
            tio.Clamp(out_min=-150, out_max=1000, exclude=['irm']),
            tio.RescaleIntensity(out_min_max=(-1, 1)),
            tio.RandomFlip(axes=('lr',), p=0.5)  # Flip left right with a probability of 0.5
        ])

        for patient_number in sorted(os.listdir(root_dir), key=lambda x: int(x)):
            patient_dir = os.path.join(root_dir, patient_number)
            if not os.path.isdir(patient_dir):
                continue

            irm_path = os.path.join(patient_dir, 'MRI', patient_number + '_valid_slices.nii')
            ct_path = os.path.join(patient_dir, 'CT', patient_number + '_valid_slices.nii')
            if os.path.exists(irm_path) and os.path.exists(ct_path):
                subject = tio.Subject(
                    ct=tio.ScalarImage(ct_path),
                    irm=tio.ScalarImage(irm_path),
                    name=patient_number,
                )
                subjects.append(subject)

        super().__init__(subjects, transform=transform)

        # Create a sampler for generating patches
        sampler = tio.data.UniformSampler(patch_size)

        # Queue that loads and returns patches
        self.patches_queue = tio.Queue(
            self,
            max_length=600,  # Maximum number of patches that can be stored in the queue
            samples_per_volume=8,
            sampler=sampler,
            num_workers=92
        )


