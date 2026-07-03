import time
import numpy as np
from PIL import Image

image = Image.open("stonehenge-colored.jpg")
grayscale_image = image.convert('L')

matrix = np.array(grayscale_image, dtype=float)

# ----- SVD ------
U, S, V = np.linalg.svd(matrix, full_matrices=False)

ranks_to_show = [1, 10, 50, 150, 300, 1000, matrix.shape[0]]

for k in ranks_to_show:
    # Efficient matrix reconstruction using slicing: U_k * Sigma_k * V_k
    # S[:k] gets the top k singular values, np.diag makes it a matrix
    approximated_matrix = U[:, :k] @ np.diag(S[:k]) @ V[:k, :]

    approximated_matrix = np.clip(approximated_matrix, 0, 255).astype(np.uint8)

    approximated_image = Image.fromarray(approximated_matrix, mode='L')

    # Save or show
    print(f"Showing approximation with top {k} singular values")
    approximated_image.show()
    time.sleep(2)