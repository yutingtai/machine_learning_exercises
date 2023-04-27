from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np



iris_dataset = pd.read_csv('Iris_dataset.csv')
X = pd.DataFrame(np.c_[iris_dataset['sepal_length'], iris_dataset['sepal_width'], iris_dataset['petal_length'],iris_dataset['petal_width']], columns=['sepal_length', 'sepal_wideth','petal_length','petal_width'])
y = iris_dataset['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# ---------- Gaussian NB model --------------#
model_gaussian = GaussianNB()
model_gaussian.fit(X_train,y_train)
model_gaussian.predict(X_test)
# model_gaussian.predict_proba(X_test)
print(model_gaussian.score(X_test,y_test))

# -------------- MultinomialNB --------------#
model_multinomial = MultinomialNB()
model_multinomial.fit(X_train, y_train)
model_multinomial.predict(X_test)
print(model_multinomial.score(X_test,y_test))




