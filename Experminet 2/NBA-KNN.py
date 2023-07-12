import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
data = pd.read_csv('Nba Player Salaries.csv')
data = data.dropna()
features = data[['2022/2023', '2023/2024', '2024/2025']]
player_names = data['Player Name']
target = data['2K-Rating']
features = features.replace({'\$': '', ',': ''}, regex=True).astype(float)
X_train, X_test, y_train, y_test, names_train, names_test = train_test_split(features, target, player_names, test_size=0.5, random_state=42)
n_neighbors_values = range(1, 21)
best_mse = float('inf')
best_n_neighbors = None
for n_neighbors in n_neighbors_values:
    model = KNeighborsRegressor(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"n_neighbors: {n_neighbors}, MSE: {mse}")
    if mse < best_mse:
        best_mse = mse
        best_n_neighbors = n_neighbors
print("Best n_neighbors:", best_n_neighbors)
print("Best MSE:", best_mse)
