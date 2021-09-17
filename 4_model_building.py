import pandas as pd
import numpy as np
import pickle
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor

def find_best_model_using_gridsearchcv(X, y):
    algos = {
        'linear_regression': {
            'model': LinearRegression(),
            'params': {
                'normalize': [True, False]
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1,2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion': ['mse', 'friedman_mse'],
                'splitter': ['best', 'random']
            }
        }
    }
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs = GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(X,y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })

    return pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])

def predict_price(location, sqft, bath, bhk):
    loc_index = np.where(X.columns == location)[0][0]
    # print(loc_index, type(loc_index))

    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return lr_clf.predict([x])[0]

df = pd.read_csv('data_file_cleaned_feateng_outrem.csv')
print(df.head())
print(df.shape)

# The location variable is textual, but needs to be numeric for model training
# You can use one hot encoding or dummy variables

dummies = pd.get_dummies(df.location)

df2 = pd.concat([df, dummies.drop('other', axis = 'columns')], axis = 'columns')

# Remember to avoid dummy variable trap, we need to drop one column (in this case 'other')

print(df2.head())

# Now define separate your features from your target

X = df2.drop(['location', 'price'], axis = 'columns')

print(df.total_sqft)
print(df.isnull().sum())

y = df2.price

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=10)

lr_clf = LinearRegression()
lr_clf.fit(X_train, y_train)
print(lr_clf.score(X_test, y_test))

# I'm getting a score of 65% which isn't that great
# In practise, we try multiple models and see what works
# We can do a k-fold cross validation

cv = ShuffleSplit(n_splits=5, test_size = 0.2, random_state=0)
cv_scores = cross_val_score(LinearRegression(), X, y, cv=cv)
print(cv_scores)

# What about other regression techniques?
# Here, we need a gridsearch cv (in the massive function at the top)

resultant = find_best_model_using_gridsearchcv(X, y)
print(resultant)

# I wonder if this can be improved by keeping the price in rupees

print(predict_price('1st Phase JP Nagar', 1000, 2, 2))
print(predict_price('1st Phase JP Nagar', 1000, 3, 3))
print(predict_price('Indira Nagar', 1000, 2, 2))
print(predict_price('Indira Nagar', 1000, 3, 3))

# Now we can export the data by pickling
# We also need the column index from our encoding

with open('bangalore_home_prices_model.pickle', 'wb') as f:
    pickle.dump(lr_clf, f)

columns = {
    'data_columns': [col.lower() for col in X.columns]
}

with open('columns.json', 'w') as f:
    f.write(json.dumps(columns))
