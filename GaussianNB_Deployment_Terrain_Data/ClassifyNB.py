from sklearn.naive_bayes import GaussianNB

def classify(features_test, labels_train):
	clf = GaussianNB()
	clf.fit(features_test, labels_train)
	pred = clf.predict(features_test)

	return clf