import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
data = pd.read_csv('Nba Player Salaries.csv')
data = data.dropna()
features = data[['2022/2023', '2023/2024', '2024/2025']]
player_names = data['Player Name']
target = data['2K-Rating']
features = features.replace({'\$': '', ',': ''}, regex=True).astype(float)
X_train, X_test, y_train, y_test, names_train, names_test = train_test_split(features, target, player_names, test_size=0.5, random_state=42)
n_estimators_values = range(50, 101)
max_depth_values = range(1, 6)
best_mse = float('inf')  
best_n_estimators = None
best_max_depth = None
for n_estimators in n_estimators_values:
    for max_depth in max_depth_values:
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        if mse < best_mse:
            best_mse = mse
            best_n_estimators = n_estimators
            best_max_depth = max_depth
print("Best Mean Squared Error:", best_mse)
print("Corresponding n_estimators:", best_n_estimators)
print("Corresponding max_depth:", best_max_depth)
