import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


music_data = pd.read_csv('music.csv')
X = music_data.drop(columns=['genre']).values
y = music_data['genre']
# train_test_split function return tuple, can be unpacked
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

model = DecisionTreeClassifier()
model = model.fit(X_train, y_train)
predictions = model.predict(X_test)

score = accuracy_score(y_test, predictions)

print(predictions)
print(score)
