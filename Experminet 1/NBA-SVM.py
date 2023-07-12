import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

# Read the CSV file
data = pd.read_csv('Nba Player Salaries.csv')

# Drop rows with missing values
data = data.dropna()

# Extract the features (salaries), player names, and target variable (ratings)
features = data[['2022/2023', '2023/2024', '2024/2025']]
player_names = data['Player Name']
target = data['2K-Rating']

# Preprocess the salary values
features = features.replace({'\$': '', ',': ''}, regex=True).astype(float)

# Split the data into training and testing datasets
X_train, X_test, y_train, y_test, names_train, names_test = train_test_split(features, target, player_names, test_size=0.5, random_state=42)

# Train the SVM model
model = SVR()
model.fit(X_train, y_train)

# Predict ratings for the test dataset
y_pred = model.predict(X_test)

# Print predicted and actual ratings with player names for each player in the test dataset
for i in range(len(X_test)):
    player_name = names_test.values[i]
    predicted_rating = y_pred[i]
    actual_rating = y_test.values[i]
    print(f"Player: {player_name}\n  Predicted Rating = {predicted_rating:.2f}\n  Actual Rating = {actual_rating}\n")

# Predict ratings for new players
new_players = pd.DataFrame({'2022/2023': [15000000, 2500000],
                            '2023/2024': [20000000, 4000000],
                            '2024/2025': [25000000, 10000000]})

predicted_ratings = model.predict(new_players)
player_names = ['Arik Tatievski', 'Roi Meshulam']

# Print predicted ratings for new players
for i in range(len(new_players)):
    player_name = player_names[i]
    predicted_rating = predicted_ratings[i]
    print(f"Player: {player_name}\n  Predicted Rating = {predicted_rating:.2f}\n")

# Calculate the difference between predicted and actual ratings
rating_difference = y_pred - y_test.values

# Convert the rating differences to integers
rating_differences_int = [int(rd) for rd in rating_difference]

# Plot the histogram
plt.hist(rating_differences_int, bins='auto')
plt.xlabel('Rating Difference')
plt.ylabel('Frequency')
plt.title('Distribution of Rating Differences')
plt.grid(True)
plt.show()
