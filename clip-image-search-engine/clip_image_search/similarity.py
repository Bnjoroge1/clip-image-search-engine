from sklearn.neighbors import KNeighborsRegressor
import numpy as np
import torch

def calculate_similarity(embedding1, embedding2, k=2):
    embedding1 = torch.Tensor(embedding1)
    embedding2 = torch.Tensor(embedding2)

    # Reshape embeddings if needed
    if len(embedding1.shape) == 1:
        embedding1 = embedding1.unsqueeze(0)
    if len(embedding2.shape) == 1:
        embedding2 = embedding2.unsqueeze(0)

    # Ensure both embeddings have the same number of samples
    max_samples = max(embedding1.shape[0], embedding2.shape[0])
    if embedding1.shape[0] < max_samples:
        embedding1 = torch.cat([embedding1] * (max_samples // embedding1.shape[0]) + [embedding1[:max_samples % embedding1.shape[0]]])
    elif embedding2.shape[0] < max_samples:
        embedding2 = torch.cat([embedding2] * (max_samples // embedding2.shape[0]) + [embedding2[:max_samples % embedding2.shape[0]]])

    # Combine embeddings into a single array
    combined_embeddings = np.vstack((embedding1.detach().numpy(), embedding2.detach().numpy()))

    # Create KNN model
    knn_model = KNeighborsRegressor(n_neighbors=k)
    
    # Fit the KNN model
    knn_model.fit(combined_embeddings, [0] * embedding1.shape[0] + [1] * embedding2.shape[0])

    # Predict distances
    distances, indices = knn_model.kneighbors(combined_embeddings, n_neighbors=k)

    # Calculate similarity score as inverse of the mean distance if mean distance is not zero
    mean_distance = np.mean(distances)
    similarity_score = 1 / mean_distance if mean_distance != 0 else 0.0
    return similarity_score