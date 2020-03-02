import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

  
pd.options.mode.chained_assignment = None


def process_df(path):
	df2 = pd.read_csv(path)
	#df2.dropna(axis=0, subset=['Age', "Embarked"], inplace=True)
	df2.Embarked.fillna("S", inplace=True)
	mean_age = df2.Age.mean()
	df2.Age.fillna(mean_age, inplace=True)

	df2["Title"] = df2.Name.str.split(",").str.get(1).str.split(" ").str.get(1).str.replace(".", "")
	df2["Title"] = df2.Title.apply(lambda x: x if x in ['Mr', 'Miss', 'Mrs', 'Master'] else 'Other')
	enc = OneHotEncoder()
	dftitle = pd.DataFrame(enc.fit_transform(df2[["Title"]]).toarray(), index=df2.index)
	df2[['Master', 'Miss', 'Mr', 'Mrs', 'Other']] = dftitle
	dfembarked = pd.DataFrame(enc.fit_transform(df2[["Embarked"]]).toarray(), index=df2.index)
	df2[["Embarked_C", "Embarked_Q", "Embarked_S"]] = dfembarked
	ordinal_enc = OrdinalEncoder()
	df2["Sex2"] = ordinal_enc.fit_transform(df2[["Sex"]])
	return df2
	

def scale_features(X):
	scaler = StandardScaler()
	X[['Age', 'Fare']] = scaler.fit_transform(X[['Age', 'Fare']])
	#poly = PolynomialFeatures(2)
	#X = poly.fit_transform(X)
	return X
	
	
def make_submission(clf, subset):
	dftest = process_df('test.csv')
	X_submit = dftest[subset]
	X_submit = scale_features(X_submit)
	predictions = clf.predict(X_submit)
	dfsubmit = pd.DataFrame({'PassengerId': dftest.PassengerId, 'Survived': predictions})
	dfsubmit.to_csv('submission.csv', index=False)
	
	
#def main():
df2 = process_df('train.csv')
subset = ['Sex2', 'Age', "Pclass", 'SibSp','Fare', "Embarked_C", "Embarked_Q", "Embarked_S",
 'Mr', 'Miss', 'Mrs', 'Master', 'Other', 'Parch']
X = df2[subset]
y = df2.Survived
X = scale_features(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

# #clf = LogisticRegression(solver="lbfgs", max_iter=200)
# #parameters = {'C':[0.01,0.03,0.1,0.3,1,3,10,30]}
parameters = {'C':[0.01,0.03,0.1,0.3,1,3,10,30],
'gamma':[0.01,0.03,0.1,0.3,1,3,10,30]}	
clf = SVC(gamma="auto", kernel="rbf")
clf = GridSearchCV(clf, parameters, cv=5, n_jobs=-1)
clf.fit(X_train, y_train)

test_score = clf.score(X_test, y_test)
train_score = clf.score(X_train, y_train)
print("Train score:", train_score, "Test score:", test_score)
make_submission(clf, subset)
	
	
# This is the standard boilerplate that calls the main() function.
#if __name__ == '__main__':
#	main()