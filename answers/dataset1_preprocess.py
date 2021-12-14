#removing rows with missing values
food_data = food_data.dropna()

#separating features and the target
X = food_data.drop(columns=['title','vegetarian','vegan','salad','side'])
y = food_data['vegetarian']

#removing indicator features with less than 2% of True instances
bad_columns=[]
for column in X.columns:
    if set(X[column].values)=={0.0, 1.0}:
        if np.mean(X[column].values)<0.02:
            bad_columns.append(column)

X = X.drop(columns=bad_columns)