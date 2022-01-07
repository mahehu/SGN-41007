from sklearn.linear_model import LogisticRegression, RandomizedLogisticRegression
from sklearn.metrics import accuracy_score
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

	lr = RandomizedLogisticRegression()
	lr.fit(X_train, y_train.ravel())

	coef = lr.get_support()
	nz = np.nonzero(coef)[0]

	print("Num_coeff: %4d" % \
		   (np.count_nonzero(coef)))

	x = X_train[0,:]
	
	plt.plot(x)
	plt.plot(nz, x[nz], 'ro')
	plt.show()
	