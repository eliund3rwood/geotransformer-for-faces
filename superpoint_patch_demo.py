import torch
import numpy as np
from sklearn.decomposition import PCA
from neighborhoods_new import neighborhoods, visualize_perturbed_pcd

# --- Configuration ---
SRC = "data/UHM_downsampled/train" 
SUPERPOINT_INDICES = [75, 411, 2699, 911, 8594, 3380, 6731, 9710, 9633, 119]
K_NEIGHBORS = 700                
SAMPLE_IDX = 0                  
N_PCA_COMPONENTS = 100         # If this is >= (K_NEIGHBORS * 3), reconstruction is perfect

arr, pcd_paths, col_means_list, _ = neighborhoods(
    src=SRC, 
    superpoint_indices=SUPERPOINT_INDICES, 
    k_neighbors=K_NEIGHBORS
)


print(f"--- Reconstructing Superpoint: {SUPERPOINT_INDICES[4]} ---")

# 1. Get original centered neighborhood
X = arr[4] 

# 2. Fit PCA
pca = PCA(n_components=min(N_PCA_COMPONENTS, X.shape[1]))
Z = pca.fit_transform(X)

# 3. Project back WITHOUT scaling
# We take the encoded 'Z' and immediately turn it back into coordinate space
reconstructed = pca.inverse_transform(Z[SAMPLE_IDX].reshape(1, -1)) 

# 4. Add the mean back to return to world coordinates
col_mean = col_means_list[4]          
reconstructed += col_mean                     

# 5. Reshape for visualization
k = reconstructed.size // 3
reconstructed_patch = reconstructed.reshape((k, 3))

# 6. Visualization
ply_path = pcd_paths[SAMPLE_IDX]
current_sp_idx = SUPERPOINT_INDICES[4]

# In the viewer:
# Blue = Original Full Point Cloud
# Red = The PCA-reconstructed patch
visualize_perturbed_pcd(ply_path, current_sp_idx, reconstructed_patch)