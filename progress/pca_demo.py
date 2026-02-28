from sklearn.decomposition import PCA
from neighborhoods_new import neighborhoods, visualize_perturbed_pcd


SRC = "UHM_downsampled/test" 
SUPERPOINT_INDICES = [8748, 2320, 10095, 2103, 8850, 6888, 8985, 8562, 4607, 1860]
K_NEIGHBORS = 700                
SUPERPOINT_IDX = 2         # which superpoint neighborhood to modify
SAMPLE_IDX = 0                # which sampled pcd to modify
N_PCA_COMPONENTS = 2000         # num PCA components

# Customizable Perturbation Params:
SCALE = 1.5                 # multiply selected PCA components by this factor
COMPONENTS_TO_SCALE = [0, 1]  # list of components to scale (0 = first principal component)


arr, pcd_paths, col_means_list = neighborhoods(SRC, SUPERPOINT_INDICES, K_NEIGHBORS)

# PCA
X = arr[SUPERPOINT_IDX]       # shape (n_samples, k*3), already mean-centered
pca = PCA(n_components=min(N_PCA_COMPONENTS, X.shape[1]))
Z = pca.fit_transform(X)

# perturb neighbors in PCA space
z = Z[SAMPLE_IDX].copy()
for c in COMPONENTS_TO_SCALE:
    z[c] *= SCALE

# project back into original coord space and add back col means
perturbed_neighbors = pca.inverse_transform(z.reshape(1,-1)) 
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
