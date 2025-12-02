import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds


ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies.csv')

ratings_matrix = ratings.pivot(index='userId', columns='movieId', values='rating')

print(f"Data before filtering:\n{ratings_matrix}\n")

ratings_filtered_matrix = ratings_matrix.dropna(thresh = 30, axis = 0)
ratings_filtered_matrix = ratings_filtered_matrix.dropna(thresh = 20, axis = 1)

print(f"Data after filtering(before the prediction):\n{ratings_filtered_matrix}\n")

ratings_matrix_filled = ratings_filtered_matrix.fillna(2.5)
R = ratings_matrix_filled.values
user_ratings_mean = np.mean(R, axis = 1)
R_demeaned = R - user_ratings_mean.reshape(-1, 1)

U, sigma, Vt = svds(R_demeaned, k = 60)
sigma = np.diag(sigma)

all_user_predicted_ratings = np.dot (np.dot (U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
preds_df = pd.DataFrame(all_user_predicted_ratings,
columns=ratings_filtered_matrix.columns, index=ratings_filtered_matrix.index)

print(f"Data after prediction:\n{preds_df}\n")

preds_only_df = preds_df.copy()
preds_only_df[ratings_filtered_matrix.notna()] = np.nan

print(f"Table with predicted ratings only:\n{preds_only_df}\n")

def recommendations(user_id, number_of_recommendations):
    prediction_for_user = preds_only_df.loc[user_id]
    sorted_prediction_for_user = prediction_for_user.dropna().sort_values(ascending = False)
    best_recomandations = sorted_prediction_for_user.head(number_of_recommendations)
    recommendations_table = pd.DataFrame(best_recomandations)
    recommendations_table.columns = ['Prediction']
    final = recommendations_table.merge(movies, how = 'left', on = 'movieId')
    return final[['title', 'genres', 'Prediction']]

print(f"Showing our top-10 recommendations for user: \n")
test = recommendations(500, 10)
print(test.to_string())