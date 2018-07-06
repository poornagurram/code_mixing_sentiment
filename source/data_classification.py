import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.naive_bayes import GaussianNB
from skelarn.metrics import classification_report
data = pd.read_csv("/Users/prudhvi/Downloads/code_mix_final_thu.csv")


docs = list(data['Tweet'])
vec = CountVectorizer()
X = vec.fit_transform(docs)
df = pd.DataFrame(X.toarray(),columns=vec.get_feature_names())
tfidf_transformer = TfidfTransformer()
X = tfidf_transformer.fit_transform(df)
y = data[['Neutral','Positive','Negative']].as_matrix()
y = np.argmax(y, axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=4)
model = LogisticRegression()
model.fit(X_train,y_train)
y_pr = model.predict(X_train_tfidf)
accuracy_score(y_test,y_pr)
print(classification_report(y_test,y_pr))
clf = GaussianNB()
clf.fit(X_train,y_train)
clf.fit(X_train.toarray(),y_train.toarray())
y_gnb = clf.predict(X_test)
accuracy_score(y_test,y_gnb)
print(classification_report(y_test,y_pr))