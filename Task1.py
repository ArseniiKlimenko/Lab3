import numpy as np

def svd(A):
    AT_A = A.T @ A
    eigenvalues, V = np.linalg.eig(AT_A)
    priority = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[priority]
    V = V[:, priority]
    singular_values = np.sqrt(np.abs(eigenvalues))
    without_noise = singular_values > 1e-5
    singular_values = singular_values[without_noise]
    V = V[:, without_noise]

    U_set = []
    for i in range(len(singular_values)):
        v_vector = V[:, i]
        sigma = singular_values[i]
        u_vector = (A @ v_vector) / sigma
        U_set.append(u_vector)

    U = np.array(U_set).T
    Sigma = np.diag(singular_values)
    Vt = V.T

    return  U, Sigma, Vt

matrix = np.array([
    [2, 2, 4],
    [3, -2, 5]
])

print(f"Original matrix:\n{matrix}\n")
U, SIGMA, Vt = svd(matrix)
print(f"Matrices we got from SVD: \n")
print(f"U:\n{U}\n")
print(f"Sigma(diagonal matrix):\n{SIGMA}\n")
print(f"V.T:\n{Vt}\n")

original_matrix = (U @ SIGMA) @ Vt
print(f"Renewed matrix:\n{original_matrix}")

if np.allclose(matrix, original_matrix):
    print("Matrices are equal! Everything is working!")
else:
    print("Error")
