from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import pandas as pd

#Load DataSet
df = pd.read_csv("heart.csv", na_values="?")

#Drop Rows with Non Number Values
df = df.dropna()

#Shows DataSets
print(df)

#Split Trian and Test Data
X = df.drop(columns=["target"], axis=1).values
Y = df["target"].values

n_trian = int(len(X) * 0.9)

x_train = X[:n_trian]
y_train = Y[:n_trian]

x_test = X[n_trian:]
y_test = Y[n_trian:]

#Fit Model
clf = BernoulliNB(alpha=1.0, fit_prior=True)
clf.fit(x_train, y_train)

#Show Score
print("Score: ", clf.score(x_test, y_test) * 100, "%")

#Show Classification Report
pred = clf.predict(x_test)
print(classification_report(y_test, pred))