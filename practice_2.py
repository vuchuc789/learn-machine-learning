import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# --- 1. Load MNIST Data ---
try:
    with np.load("mnist.npz") as ds:
        # Using x_test for a smaller dataset for faster K-Means demonstration
        # images = ds["x_test"]  # Original 28x28 images for display
        # true_labels = ds["y_test"]  # Original integer labels
        # To use the full training set (might be slow for K-Means):
        # images = ds["x_train"]
        # true_labels = ds["y_train"]
        images = np.concatenate((ds["x_train"], ds["x_test"]), axis=0)
        true_labels = np.concatenate((ds["y_train"], ds["y_test"]), axis=0)
        shuffled_indices = np.random.permutation(len(images))
        images = images[shuffled_indices]
        true_labels = true_labels[shuffled_indices]

except FileNotFoundError:
    print(
        "Error: mnist.npz not found. Please download it or ensure it's in the correct path."
    )
    print("Using dummy data instead for demonstration.")
    images = np.random.rand(1000, 28, 28) * 255  # 1000 dummy images
    true_labels = np.random.randint(0, 10, 1000)  # Dummy labels


# --- 2. Prepare the Data for K-Means ---
num_images = images.shape[0]
# Flatten the images for K-Means input
flattened_images = images.reshape(num_images, -1)
# Normalize pixel values
flattened_images = flattened_images / 255.0

print(f"Data shape for K-Means: {flattened_images.shape}")

# --- 3. Apply K-Means Clustering ---
num_clusters = 10
kmeans = KMeans(n_clusters=num_clusters, n_init="auto", random_state=42)

print("Fitting K-Means... (This might take a moment depending on data size)")
kmeans.fit(flattened_images)
cluster_labels = kmeans.labels_
centroids = kmeans.cluster_centers_
print("K-Means clustering complete.")

# --- 4. Visualize the Centroids (as before) ---
plt.figure(figsize=(10, 4))
plt.suptitle("K-Means Cluster Centroids", fontsize=16)
for i in range(num_clusters):
    plt.subplot(2, 5, i + 1)
    centroid_image = centroids[i].reshape(28, 28)
    plt.imshow(centroid_image, cmap=plt.cm.binary)
    plt.title(f"Centroid {i}")
    plt.xticks([])
    plt.yticks([])
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# --- 5. Evaluate Clustering (as before) ---
ari = adjusted_rand_score(true_labels, cluster_labels)
nmi = normalized_mutual_info_score(true_labels, cluster_labels)
print(f"\nAdjusted Rand Index (ARI): {ari:.3f}")
print(f"Normalized Mutual Information (NMI): {nmi:.3f}")


# --- 6. NEW: Show Sample Images from Each Cluster ---
print("\nDisplaying sample images from each cluster...")
num_examples_to_show = 10  # Number of example images to show per cluster

for i in range(num_clusters):
    # Find indices of images belonging to the current cluster
    cluster_i_indices = np.where(cluster_labels == i)[0]

    if len(cluster_i_indices) == 0:
        print(f"Cluster {i} has no samples assigned to it.")
        continue

    # Randomly select 'num_examples_to_show' images from this cluster
    # If fewer images in cluster than we want to show, show all of them
    actual_num_to_show = min(num_examples_to_show, len(cluster_i_indices))
    selected_indices = np.random.choice(
        cluster_i_indices, size=actual_num_to_show, replace=False
    )  # No replacement

    plt.figure(figsize=(12, 3))  # Adjust figsize as needed
    plt.suptitle(f"Cluster {i}: Sample Images (True Labels Shown)", fontsize=14)

    for plot_idx, image_idx in enumerate(selected_indices):
        # Display in one row: 1 row, actual_num_to_show columns
        plt.subplot(1, actual_num_to_show, plot_idx + 1)
        # 'images' holds the original 28x28 images
        plt.imshow(images[image_idx], cmap=plt.cm.binary)
        # Add the true label as the title for each image
        plt.title(f"True: {true_labels[image_idx]}")
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout(rect=[0, 0, 1, 0.90])  # Adjust layout to make space for suptitle
    plt.show()
