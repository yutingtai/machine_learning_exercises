from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv('spam.csv')
df = df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'])
df = df.rename(columns={'v1': 'Label', 'v2': 'Content'})
X = df['Content']
y = df['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=True)
text_collection = Pipeline(steps=[('vectorize', CountVectorizer(ngram_range=(1, 1), token_pattern=r'\b\w+\b')),
                                  ('classifier', MultinomialNB())])
text_collection.fit(X_train, y_train)
y_predict = text_collection.predict(X_test)
print(accuracy_score(y_test, y_predict))
print(classification_report(y_test, y_predict))
mat = confusion_matrix(y_test, y_predict)
sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=True, cmap='cool')
plt.xlabel('predicted value')
plt.ylabel('true value')
plt.show()


# def split_data_for_training(X, y):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=True)
#     return X_train, X_test, y_train, y_test
#
# def text_collection_model():
