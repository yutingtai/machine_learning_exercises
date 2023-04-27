import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


boston_dataset = pd.read_csv('boston_dataset.csv', skiprows=1)

# check the dataset isn't lack of any information
print(boston_dataset.isnull().sum())

sns.set(rc={'figure.figsize': (10, 10)})
sns.displot(boston_dataset['MEDV'])
correlation_matrix = boston_dataset.corr().round(2)
# annot = True , put the number in the cells
plt.figure(figsize=(15, 5))
sns.heatmap(data=correlation_matrix, annot=True)
plt.show()

X = pd.DataFrame(np.c_[boston_dataset['LSTAT'], boston_dataset['RM']], columns=['LSTAT', 'RM'])
y = boston_dataset['MEDV']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

reg = LinearRegression()
model = reg.fit(X_train, y_train)
predictions = reg.predict(X_test)
coeff_df = pd.DataFrame(reg.coef_, X_train.columns, columns=['Coefficient'])
print(coeff_df)
print(reg.intercept_)
print('R2: ', reg.score(X_test, y_test))
