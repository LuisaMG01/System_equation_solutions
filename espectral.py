import numpy as np

# Define the matrix A
A = np.array([[3, -5, -8], 
              [2, 4, 6], 
              [3, 4, -12]], dtype=float)
# Calculate the eigenvalues
eigenvalues = np.linalg.eigvals(A)

# Calculate the spectral radius
spectral_radius = max(abs(eigenvalues))

print("Eigenvalues:", eigenvalues)
print("Spectral Radius:", spectral_radius)
