import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor


url = "https://raw.githubusercontent.com/leontoddjohnson/datasets/refs/heads/main/data/coffee_analysis.csv"

df_coffee = pd.read_csv(url)


# Train Model #1 Linear Regression and save model as Pickle
df_train, df_test = train_test_split(df_coffee, test_size=0.2, random_state=42)

X = df_train[["100g_USD"]]
y = df_train[["rating"]]

model_1 = LinearRegression()
model_1.fit(X, y)

with open('model_1.pickle', 'wb') as file:
    pickle.dump(model_1, file)


# Define a roast category dictionary
roast_cat = {
    'Light': 0,
    'Medium-Light': 1,
    'Medium': 2,
    'Medium-Dark': 3,
    'Dark': 4
}

# Save roast category dictionary to pickle
with open('roast_cat.pickle', 'wb') as file:
    pickle.dump(roast_cat, file)

df_coffee['roast_num'] = df_coffee['roast'].map(roast_cat)

# Recreate Test/Train Split (honestly, the directions don't even mention doing a split but...)
df_train, df_test = train_test_split(df_coffee, test_size=0.2, random_state=42)

# Train Model #2 Decision Tree and save model as Pickle
X = df_train[["100g_USD", "roast_num"]]
y = df_train['rating']

model_2 = DecisionTreeRegressor()
model_2.fit(X, y)

with open('model_2.pickle', 'wb') as file:
    pickle.dump(model_2, file)