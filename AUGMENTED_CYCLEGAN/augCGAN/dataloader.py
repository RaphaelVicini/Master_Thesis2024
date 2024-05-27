import torchio as tio
from torch.utils.data import DataLoader
import os
import torch

# Définition des transformations à appliquer
# ct_transform = tio.Clamp(out_min=0, out_max=2500)  # Pour les images CT
# mri_transform = tio.RescaleIntensity(out_min_max=(-1, 1))  # Pour les images MRI


# # Pipeline de transformations appliqué de façon conditionnelle
# class CustomTransform:
#     def __call__(self, subject):
#         # Appliquer le Clamp uniquement sur les images CT
#         subject['ct'].set_data(ct_transform(subject['ct'].data))

#         # Appliquer le RescaleIntensity uniquement sur les images MRI
#         subject['irm'].set_data(mri_transform(subject['irm'].data))

#         return subject



class CustomPatientDataset(tio.SubjectsDataset):
    def __init__(self, root_dir):
        subjects = []  # Cette liste sera remplie avec des objets tio.Subject

        # Définir les transformations ici
        transform = tio.Compose([
            tio.ToCanonical(),
            tio.Clamp(out_min=0, out_max=2500),  # Notez que le include doit être géré différemment
            tio.RescaleIntensity(out_min_max=(-1, 1))  # Même note pour include
        ])

        for patient_number in sorted(os.listdir(root_dir), key=lambda x: int(x)):
            patient_dir = os.path.join(root_dir, patient_number)
            if not os.path.isdir(patient_dir):
                continue

            irm_path = os.path.join(patient_dir, 'MRI')
            ct_path = os.path.join(patient_dir, 'CT')

            subject = tio.Subject(
                ct=tio.ScalarImage(ct_path),
                irm=tio.ScalarImage(irm_path),
                name=patient_number,
            )
            subjects.append(subject)

        # Appeler le constructeur de la classe parente avec la liste des sujets et la transformation
        super().__init__(subjects, transform=transform)

# Chemin vers le dossier racine contenant les dossiers des patients
root_dir = r'C:\Users\rapha\OneDrive\Bureau\Mémoire\Data\Patients'

# Création de l'instance de dataset
dataset = CustomPatientDataset(root_dir=root_dir)

# Création du DataLoader
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Exemple d'itération sur le DataLoader
for batch in dataloader:
    ct = batch['ct'][tio.DATA]
    irm = batch['irm'][tio.DATA]
    #print(ct.shape, irm.shape)

"Verification des transformations "



# Supposons que 'dataloader' est déjà créé et contient vos données

for i, batch in enumerate(dataloader):
    ct = batch['ct'][tio.DATA]
    irm = batch['irm'][tio.DATA]

    # Calculer les statistiques pour CT
    ct_min = torch.min(ct)
    ct_max = torch.max(ct)
    ct_mean = torch.mean(ct)
    ct_median = torch.median(ct)

    # Calculer les statistiques pour IRM
    irm_min = torch.min(irm)
    irm_max = torch.max(irm)
    irm_mean = torch.mean(irm)
    irm_median = torch.median(irm)

    print(f"Batch {i+1} CT Statistiques: Min={ct_min.item()}, Max={ct_max.item()}, Moyenne={ct_mean.item()}, Médiane={ct_median.item()}")
    print(f"Batch {i+1} IRM Statistiques: Min={irm_min.item()}, Max={irm_max.item()}, Moyenne={irm_mean.item()}, Médiane={irm_median.item()}\n")

    # Optionnel: Arrêter après le premier batch pour l'exemple
    if i == 0: break
