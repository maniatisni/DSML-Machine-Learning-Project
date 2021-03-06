import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge, LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor

wandb.init(project="visualize-sklearn")
wandb.run.name = "linear"

features = ['freq', 'angle', 'length', 'velocity', 'thickness', 'pressure']
df = pd.read_table('./datasets/airfoil_self_noise.dat', names=features)

X = df.copy()
X = X.drop(['pressure'], axis=1)
y = df['pressure']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=75)

# reg = Ridge()
# reg.fit(X, y)
#
# wandb.sklearn.plot_regressor(reg, X_train, X_test, y_train, y_test, 'Ridge')

lr = LinearRegression()
lr.fit(X, y)

wandb.sklearn.plot_regressor(lr, X_train, X_test, y_train, y_test, 'linear regression')
