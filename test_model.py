from time import time

import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score

tfidf = TfidfVectorizer(stop_words='english', strip_accents='ascii')
personalty = pd.read_csv('data/mypersonality_final.csv', encoding="ISO-8859-1")
d = {'y': True, 'n': False}

x = personalty['STATUS']
X_bar = tfidf.fit_transform(x)
y_sbar = personalty['sOPN']
y_cbar = personalty['cOPN'].map(d)

X_strain, X_stest, y_strain, y_stest = train_test_split(X_bar, y_sbar, test_size=0.2)
X_ctrain, X_ctest, y_ctrain, y_ctest = train_test_split(X_bar, y_cbar, test_size=0.2)

#bar = RandomForestRegressor(n_estimators=300, max_depth=8)
# bar = RandomForestRegressor(bootstrap=True, max_features='sqrt', min_samples_leaf=1, min_samples_split=2,
#                           n_estimators=200)
# start_time = time()
# bar.fit(X_strain, y_strain)
# elapsed_time = time() - start_time
# print("Elapsed time: %.10f seconds." % elapsed_time)
# Y_pred = bar.predict(X_stest)
# print(f'Precision del  modelo {bar.score(X_strain, y_strain)}')

bar = RandomForestClassifier(max_features='sqrt', n_estimators=110)

start_time = time()
bar.fit(X_ctrain, y_ctrain)
elapsed_time = time() - start_time
print("Elapsed time: %.10f seconds." % elapsed_time)
Y_pred = bar.predict(X_ctest)
print(f'Precision del  modelo {precision_score(y_ctest, Y_pred)}')

matrix = confusion_matrix(y_ctrain, Y_pred)
print(f'Precision del  modelo {matrix}')