from sklearn.linear_model import LogisticRegression, RandomizedLogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Normalizer
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

	plt.style.use('classic')
	
	D = loadmat("arcene.mat")

	X_test = D["X_test"]
	X_train = D["X_train"]
	y_test = D["y_test"].ravel()
	y_train = D["y_train"].ravel()
	
	normalizer = Normalizer()
	normalizer.fit(X_train)
	X_train = normalizer.transform(X_train)
	X_test  = normalizer.transform(X_test)	

	C_range = 10.0 ** np.linspace(-5,8,10)
	scores = []
	
	for C in C_range:

		lr = LogisticRegression(penalty = 'l2', C = C)
		lr.fit(X_train, y_train.ravel())

		y_pred = lr.predict(X_test)
		accuracy = accuracy_score(y_pred, y_test)
		
		print("C = %11.1f: num_coeff: %4d, norm = %.4f, acc = %.2f %%" % \
		       (C,
			    np.count_nonzero(lr.coef_),
				np.linalg.norm(lr.coef_, ord = 1),
				100 * accuracy))

		scores.append(accuracy)
				
	best_C = C_range[np.argmax(scores)]
	print("Using C = %.2f" % best_C)
	
	lr = LogisticRegression(penalty = 'l2', C = best_C)
	lr.fit(X_train, y_train.ravel())

	coef = lr.coef_.ravel()
	nz = np.nonzero(coef)[0]
	
	x = X_train[0,:]
	
	plt.plot(x)
	plt.plot(nz, x[nz], 'ro')
	plt.show()
	