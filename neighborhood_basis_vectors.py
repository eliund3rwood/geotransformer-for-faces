import torch
import numpy as np
from sklearn.decomposition import PCA
from neighborhoods_new import neighborhoods, visualize_perturbed_pcd


SRC = "data/UHM_downsampled/train" 
SUPERPOINT_INDICES = [75, 411, 2699, 911, 8594, 3380, 6731, 9710, 9633, 119, 
                3441, 6319, 9541, 8732, 6162, 3774, 8296, 3151, 10, 
                7720, 6858, 7409, 7531, 3504, 6937, 4189, 8891, 3721, 
                9241, 2213, 1765, 7547]
K_NEIGHBORS = 700                
N_PCA_COMPONENTS = 100         # Num PCA components


arr, pcd_paths, col_means_list, patch_indices = neighborhoods(src=SRC, 
                                               superpoint_indices=SUPERPOINT_INDICES, 
                                               k_neighbors=K_NEIGHBORS)

all_bases = []
all_means = []
all_z_values = [] # Store the GT coefficients here

for i in range(len(SUPERPOINT_INDICES)):
    X = arr[i] # Shape (n_samples, k*3)
    pca = PCA(n_components=N_PCA_COMPONENTS)
    Z = pca.fit_transform(X) # Projecting data into PCA space
    
    all_bases.append(pca.components_) 
    all_means.append(col_means_list[i])
    all_z_values.append(Z) # Shape (n_samples, 100)

stacked_basis = torch.from_numpy(np.stack(all_bases)).float()
stacked_means = torch.from_numpy(np.stack(all_means)).float()
# Shape: [num_superpoints, num_samples, 100]
stacked_z = torch.from_numpy(np.stack(all_z_values)).float()

torch.save({
    'basis': stacked_basis,
    'mean': stacked_means,
    'gt_z': stacked_z, # Pre-calculated coefficients
    'patch_indices': torch.from_numpy(patch_indices).long()
}, "pca_basis_all.pth")