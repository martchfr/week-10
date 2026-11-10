import pandas as pd
import pickle

# Load trained models
with open('model_1.pickle', 'rb') as file:
    model_1 = pickle.load(file)

with open('model_2.pickle', 'rb') as file:
    model_2 = pickle.load(file)

# Load roast dictionary
with open('roast_cat.pickle', 'rb') as file:
    roast_cat = pickle.load(file)


def predict_rating(df_X):

    predictions = []

    for index, row in df_X.iterrows():

        # Check whether roast category exists
        if row['roast'] in roast_cat:

            # Convert roast to numerical value
            roast_num = roast_cat[row['roast']]

            # Use both features
            X = pd.DataFrame({
                '100g_USD': [row['100g_USD']],
                'roast_num': [roast_num]
            })

            prediction = model_2.predict(X)[0]

        else:

            # Use price only
            X = pd.DataFrame({
                '100g_USD': [row['100g_USD']]
            })

            prediction = model_1.predict(X)[0]

        predictions.append(prediction)

    return predictions