import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

music_data = pd.read_csv('music.csv')
X = music_data.drop(columns=['genre']).values
y = music_data['genre']

model = DecisionTreeClassifier()
model.fit(X, y)

dot_data = tree.export_graphviz(model, out_file='music-recommender.dot',
                                feature_names=['age', 'gender'],
                                class_names=sorted(y.unique()),
                                label='all',
                                rounded=True,
                                filled=True)

plt.figure()
plot_tree(model,feature_names=['age', 'gender'],
          class_names=sorted(y.unique()),
          label='all',
          rounded=True,
          filled=True)
plt.title("music-recommender")
plt.savefig('music-recommender.png')
plt.show()
