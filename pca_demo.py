import torch
from sklearn.decomposition import PCA
from neighborhoods_new import neighborhoods, visualize_perturbed_pcd


SRC = "data/UHM_downsampled/train" 
SUPERPOINT_INDICES = [75, 411, 2699, 911, 8594, 3380, 6731, 9710, 9633, 119, 
                3441, 6319, 9541, 8732, 6162, 3774, 8296, 3151, 10, 
                7720, 6858, 7409, 7531, 3504, 6937, 4189, 8891, 3721, 
                9241, 2213, 1765, 7547]
K_NEIGHBORS = 700                
SUPERPOINT_IDX = 6    # which superpoint neighborhood to modify
SAMPLE_IDX = 0                # which sampled pcd to modify
N_PCA_COMPONENTS = 30         # num PCA components

# Customizable Perturbation Params:
SCALE = 1       # multiply selected PCA components by this factor
COMPONENTS_TO_SCALE = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # list of components to scale (0 = first principal component)


arr, pcd_paths, col_means_list, _ = neighborhoods(src=SRC, 
                                               superpoint_indices=SUPERPOINT_INDICES, 
                                               k_neighbors=K_NEIGHBORS)

# PCA
X = arr[SUPERPOINT_IDX]       # shape (n_samples, k*3), already mean-centered
pca = PCA(n_components=min(N_PCA_COMPONENTS, X.shape[1]))
Z = pca.fit_transform(X)

# perturb neighbors in PCA space
z = Z[SAMPLE_IDX].copy()
for c in COMPONENTS_TO_SCALE:
    z[c] *= SCALE

# project back into original coord space and add back col means
perturbed_neighbors = pca.inverse_transform(z.reshape(1, -1)) 
col_mean = col_means_list[SUPERPOINT_IDX]         
perturbed_neighbors += col_mean                     

# reshape to (k,3)
k3 = perturbed_neighbors.size
k = k3 // 3
perturbed_neighbors = perturbed_neighbors.reshape((k, 3))

# visualization
ply_path = pcd_paths[SAMPLE_IDX]
superpoint_index_in_pcd = SUPERPOINT_INDICES[SUPERPOINT_IDX]

visualize_perturbed_pcd(ply_path, superpoint_index_in_pcd, perturbed_neighbors)



basis = torch.from_numpy(pca.components_).float()
mean = torch.from_numpy(col_mean.reshape(-1)).float()

