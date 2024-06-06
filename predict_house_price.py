import pandas as pd
from datetime import datetime
from sklearn.tree import DecisionTreeRegressor


house_pricing_list = './Housing.csv'

pricing_list_pd = pd.read_csv(house_pricing_list)

# Check if DataFrame is empty
if pricing_list_pd.empty:
    print("DataFrame is empty.")
else:
    # Display first few rows
    print(pricing_list_pd.head())

    # Check for missing values
    print("Missing values:")
    print(pricing_list_pd.isnull().sum())

    # Display summary statistics for numeric columns
    print("Summary statistics:")
    print(pricing_list_pd.describe())

avg_area_of_houses = round(pricing_list_pd['area'].mean())

print(avg_area_of_houses)

pricing_list_pd = pricing_list_pd.dropna(axis=0)

print(pricing_list_pd.columns)

y = pricing_list_pd.price

pricing_features = ['area', 'bedrooms', 'bathrooms', 'parking', 'stories']

X = pricing_list_pd[pricing_features]

print(X.describe())

print(X.head())

pricing_model = DecisionTreeRegressor(random_state = 1)

pricing_model.fit(X, y)

predictions = pricing_model.predict(X)

from sklearn.metrics import mean_absolute_error
print(f"Mean Absolute Error without test-train split " + str(mean_absolute_error(y, predictions)))


# Train test split

from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

pricing_model2 = DecisionTreeRegressor(random_state = 1)

pricing_model2.fit(train_X, train_y);

prediction2 = pricing_model2.predict(val_X)

print(f"Mean Absolute Error with test-train split " + str(mean_absolute_error(val_y, prediction2)))

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y): 
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    predict_value = model.predict(val_X)
    mae = mean_absolute_error(val_y, predict_value)
    return mae

for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))

from sklearn.model_selection import cross_val_score

for max_leaf_nodes in [5, 7, 9]:
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %0.2f" % (max_leaf_nodes, -scores.mean()))

print(DecisionTreeRegressor._abc_impl)