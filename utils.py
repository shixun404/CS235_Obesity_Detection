import numpy as np


class LogisticRegression(object):
    def __init__(self, tol=0.001, max_iter=1000, penalty='norm', C=1):
        """
        self.weights weight
        self.tol threshold
        self.max_iter maximum iteration
        self.C Regularization term before coefficients
        self.penalty regularization
        """
        self.weights = None
        self.tol = tol
        self.max_iter = max_iter
        self.C = C
        self.penalty = penalty

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    # same as sklearn.fit
    def fit(self, dataMatIn, classLabels):
        dataMatrix = np.mat(dataMatIn)  # matrix
        labelMat = np.mat(classLabels)
        m, n = np.shape(dataMatrix)  # dim
        self.weights = np.ones((n, 1))  # init: 1

        for i in range(self.max_iter):
            # Assume that the function
            h = self.sigmoid(dataMatrix * self.weights)  # compress to 0-1, pos/neg
            # Regularization term, according to the gradient metric inferred, without the regularization term, L1 regular, L2 regular parameter update logic
            if self.penalty == 'norm':
                # matrix multiple (sample few, x grient down)
                self.weights = self.weights + self.tol * dataMatrix.transpose() * (labelMat - h)
            elif self.penalty == 'l1':
                # lambda
                self.weights = self.weights + self.C * self.tol * np.where(self.weights > 0, 1, -1) + self.tol * dataMatrix.transpose() * (labelMat - h)
            elif self.penalty == 'l2':
                self.weights = self.weights * (1 - self.C * self.tol) + self.tol * dataMatrix.transpose() * (labelMat - h)

    # Same role as sklearn.predict_pro, return probability
    def predict_pro(self, dataMatIn):
        dataMatrix = np.mat(dataMatIn)
        return self.sigmoid(dataMatrix * self.weights)

    # Same role as sklearn.predict, return category
    def predict(self, dataMatIn):
        dataMatrixPro = self.predict_pro(dataMatIn)
        return np.where(dataMatrixPro >= 0.5, 1, 0).reshape(len(dataMatIn), )

    # Same as sklearn.score, accuracy
    def score(self, dataMatIn, classLabels):
        yhat = self.predict(dataMatIn).reshape(len(dataMatIn), )
        y = np.array(classLabels).reshape(len(classLabels), )
        return np.sum(np.where((yhat == y) == True, 1, 0)) / len(y)


# calculate accuracy score
def accuracy_score(y, yhat):
    y = np.array(y).reshape(len(y), )
    yhat = np.array(yhat).reshape(len(yhat), )
    return np.sum(np.where((yhat == y) == True, 1, 0)) / len(y)


# calculate precision score
def precision_score(y, yhat):
    y = np.array(y).reshape(len(y), )
    yhat = np.array(yhat).reshape(len(yhat), )
    return sum(yhat * y) / np.sum(yhat)


# calculate recall score
def recall_score(y, yhat):
    y = np.array(y).reshape(len(y), )
    yhat = np.array(yhat).reshape(len(yhat), )
    return sum(yhat * y) / np.sum(y)


# calculate f1 score
def f1_score(y, yhat):
    p = precision_score(y, yhat)
    r = recall_score(y, yhat)
    return 2 * p * r / (p + r)


# calculate confusion matrix
def confusion_matrix(y, yhat):
    y = np.array(y).reshape(len(y), )
    yhat = np.array(yhat).reshape(len(yhat), )
    num1 = sum(yhat * y)
    num2 = sum(np.where(yhat == 0, 1, 0) * y)
    num3 = sum(np.where(y == 0, 1, 0) * yhat)
    num4 = sum(np.where(y == 0, 1, 0) * np.where(yhat == 0, 1, 0))
    return np.array([[num4, num3], [num2, num1]])
