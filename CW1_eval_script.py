import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
# Set seed
np.random.seed(123)

# Import training data
trn = pd.read_csv('CW1_train.csv')
X_tst = pd.read_csv('CW1_test.csv') # This does not include true outcomes (obviously)

# preprocessing:
def engineer_features(df):
    df['volume'] = df['x'] * df['y'] * df['z']
    df = df.drop(columns=['x', 'y', 'z'])
    # Impute missing values with median to ensure the script doesn't crash
    df = df.fillna(df.median(numeric_only=True))
    return df

trn = engineer_features(trn)
X_tst = engineer_features(X_tst)

# Identify categorical columns
categorical_cols = ['cut', 'color', 'clarity']

# One-hot encode categorical variables
trn = pd.get_dummies(trn, columns=categorical_cols, drop_first=True)
X_tst = pd.get_dummies(X_tst, columns=categorical_cols, drop_first=True)

# Train your model (using a simple LM here as an example)
X_trn = trn.drop(columns=['outcome'])
y_trn = trn['outcome']
model = XGBRegressor(
    n_estimators=1000, 
    max_depth=3, 
    learning_rate=0.01, 
    subsample=0.9, 
    random_state=123
)
model.fit(X_trn, y_trn)

# Test set predictions
yhat_lm = model.predict(X_tst)

# Format submission:
# This is a single-column CSV with nothing but your predictions
out = pd.DataFrame({'yhat': yhat_lm})
out.to_csv('CW1_submission_K24066415.csv', index=False) # Please use your k-number here

################################################################################

# At test time, we will use the true outcomes
tst = pd.read_csv('CW1_test_with_true_outcome.csv') # You do not have access to this

# This is the R^2 function
def r2_fn(yhat):
    eps = y_tst - yhat
    rss = np.sum(eps ** 2)
    tss = np.sum((y_tst - y_tst.mean()) ** 2)
    r2 = 1 - (rss / tss)
    return r2

# How does the linear model do?
print(r2_fn(yhat_lm))
