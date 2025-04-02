import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn

from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import mnist

class BinaryLogisticRegression:        
	def fit(self, X, y, alpha, epochs, eps, update):
		# add a column of 1s to X
		X = np.hstack((np.ones([X.shape[0],1]), X))
			
		# modify y to be n by 1
		y = np.atleast_2d(y).T
		
		# initialize the parameters to 1
		self.theta = np.ones([X.shape[1], 1])
		
		# initialize the step number and theta gradient
		step = 0
		thetagrad = 2 * eps
		
		# minimize cross-entropy -- run until thetagrad is small or step is epochs
		while np.linalg.norm(thetagrad) > eps and step < epochs:
			# compute the loss
			sig = self.sigmoid(X @ self.theta)
			
			if step % update == 0:
				loss = -(1/X.shape[0]) * np.sum(y * np.log(sig + 0.001) + (1 - y) * np.log(1 - sig + 0.001))
				print('Iteration', step, '\tLoss =', loss)
			
			# compute the gradient
			thetagrad = X.T @ (sig - y)
			
			# take a gradient descent step
			self.theta -= alpha * thetagrad
						
			# iterate the step
			step += 1
			
			if step == epochs:
				print('Gradient descent failed to converge. (The answer may still be acceptably good.)')
			
	def predict(self, X, threshold=0.5):
		# add a column of 1s to X
		X = np.hstack((np.ones([X.shape[0],1]), X))
		
		# return 0 if the posterior for Y=1 is less than for Y=0
		# otherwise, return 1
		return (self.sigmoid(X @ self.theta) >= threshold).astype(float)
			
	def sigmoid(self, z):
		z = np.clip(z, -500, 500)  # prevent overflow
		return 1 / (1 + np.exp(-z))