import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.sparse.linalg import svds
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies.csv')

ratings_matrix = ratings.pivot(index='userId', columns='movieId', values='rating')

print(f"Data before filtering:\n{ratings_matrix}\n")

ratings_filtered_matrix = ratings_matrix.dropna(thresh = 100, axis = 0)
ratings_filtered_matrix = ratings_filtered_matrix.dropna(thresh = 100, axis = 1)

print(f"Data after filtering:\n{ratings_filtered_matrix}\n")

ratings_matrix_filled = ratings_filtered_matrix.fillna(2.5)
R = ratings_matrix_filled.values
user_ratings_mean = np.mean(R, axis = 1)
R_demeaned = R - user_ratings_mean.reshape(-1, 1)

U, sigma, Vt = svds(R_demeaned, k = 3)
U = U[:, ::-1]
Vt = Vt[::-1, :]
sigma = np.diag(sigma[::-1])

print(f"Users(U):\n{U.shape}\n")
print(f"Movies(Vt):\n{Vt.shape}\n")

def visualisation(matrix, title, count=20):
    fig = plt.figure(figsize=(8,8))
    axes = fig.add_subplot(111, projection='3d')
    xs = matrix[:count, 0]
    ys = matrix[:count, 1]
    zs = matrix[:count, 2]
    axes.scatter(xs, ys, zs, marker='o', c='r')
    axes.set_title(title)
    plt.show()

visualisation(U, "Users (First 20)")
visualisation(Vt, "Films (First 20)")