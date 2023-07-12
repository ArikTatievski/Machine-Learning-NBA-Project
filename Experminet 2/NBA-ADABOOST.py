import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
data = pd.read_csv('Nba Player Salaries.csv')
data = data.dropna()
features = data[['2022/2023', '2023/2024', '2024/2025']]
player_names = data['Player Name']
target = data['2K-Rating']
features = features.replace({'\$': '', ',': ''}, regex=True).astype(float)
X_train, X_test, y_train, y_test, names_train, names_test = train_test_split(features, target, player_names, test_size=0.5, random_state=42)
learning_rate_values = [0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2]
best_mse = float('inf')
best_learning_rate = None
for learning_rate in learning_rate_values:
    model = AdaBoostRegressor(learning_rate=learning_rate)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Learning Rate: {learning_rate:.2f}, MSE: {mse}")
    if mse < best_mse:
        best_mse = mse
        best_learning_rate = learning_rate
print("Best Learning Rate:", best_learning_rate)
print("Best MSE:", best_mse)
