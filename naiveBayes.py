# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
# from naiveBayesClassifier import tokenizer
# from naiveBayesClassifier.trainer import Trainer
# from naiveBayesClassifier.classifier import Classifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

def evaluate(y_actual, y_pred):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    print('accuracy:', accuracy_score(y_actual, y_pred))
    print('precision:', precision_score(y_actual, y_pred))
    print('recall:', recall_score(y_actual, y_pred))
    print('f1:', f1_score(y_actual, y_pred))
    print('confusion matrix:')
    print(confusion_matrix(y_actual, y_pred))

class NaiveBayes(object):
    def __init__(self):
        return

    def gauss(self, test, mean, std):
        t1 = (test - mean) * (test - mean)
        t2 = std * std
        if t2 == 0:
            t2 = 0.01
        res = np.exp(-t1/(t2*2)) / np.sqrt(2*t2*np.pi)
        # return np.log(res)
        return res

    def fit (self, x_train: np.ndarray, y_train: np.ndarray):
        self.normalData = []
        self.obesityData = []
        self.norMean = []
        self.norStd = []
        self.obeMean = []
        self.obeStd = []

        for i in range(y_train.shape[0]):
            if y_train[i] == 0:
                self.normalData.append(x_train[i])
            else:
                self.obesityData.append(x_train[i])


        self.normalData = np.array(self.normalData)
        self.obesityData = np.array(self.obesityData)

        self.norMean = self.normalData.mean(axis=0)
        self.norStd = self.normalData.std(axis=0)
        self.obeMean = self.obesityData.mean(axis=0)
        self.obeStd = self.obesityData.std(axis=0)
        self.norPY = (y_train.shape[0]- np.sum(y_train)) / y_train.shape[0]
        self.obePY = np.sum(y_train)/ y_train.shape[0]

        return

    def predict(self,x_test: np.ndarray):
        y_prob = []

        for i in range(x_test.shape[0]):
            curNor = self.norPY
            curObe = self.obePY
            for j in range(x_test.shape[1]):
                curNor *= self.gauss(x_test[i,j],self.norMean[j],self.norStd[j])
                curObe *= self.gauss(x_test[i,j],self.obeMean[j],self.obeStd[j])
            y_prob.append(int(curObe > curNor))

            # pxy = np.prod(self.gauss(x_test))
        # print(y_prob)
        # print(y_test)

        return np.array(y_prob)






def preprocessing() -> pd.DataFrame:
    df = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')

    df['Label'] = df['NObeyesdad'].str.startswith('Obesity').astype(int)
    del df['NObeyesdad']
    del df['Height']
    del df['Weight']
    #
    freq_map = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}

    yes_no_map = {'yes': 1, 'no': 0}
    #
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
    df['family_history_with_overweight'] = df['family_history_with_overweight'].map(yes_no_map)
    df['FAVC'] = df['FAVC'].map(yes_no_map)
    df['CAEC'] = df['CAEC'].map(freq_map)
    df['SMOKE'] = df['SMOKE'].map(yes_no_map)
    df['SCC'] = df['SCC'].map(yes_no_map)
    df['CALC'] = df['CALC'].map(freq_map)
    df['MTRANS'] = df['MTRANS'].map(
        {'Automobile': 0, 'Motorbike': 0, 'Public_Transportation': 0, 'Walking': 1, 'Bike': 1})

    return df


# Press the green button in the gutter to run the script.


def run_naive_bayes():
    df = preprocessing()
    X = df.drop(columns=['Label'])
    y = df['Label']
    X = X.to_numpy()
    y = y.to_numpy()
    X.astype(np.int32)
    y.astype(np.int32)

    myPredAll = np.empty(shape=(0,))
    sklPredAll = np.empty(shape=(0,))
    y_test_all = np.empty(shape=(0,))

    accuracy_my_all = np.empty(shape=(0,))
    accuracy_shelf_all = np.empty(shape=(0,))

    precision_my_all = np.empty(shape=(0,))
    precision_shelf_all = np.empty(shape=(0,))

    recall_my_all = np.empty(shape=(0,))
    recall_shelf_all = np.empty(shape=(0,))

    f1_my_all = np.empty(shape=(0,))
    f1_shelf_all = np.empty(shape=(0,))

    for i in range(50):


        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i, stratify=y)
        y_test_all = np.append(y_test_all, y_test)

        nb = NaiveBayes()
        nb.fit(X_train,y_train)
        y_pred_my = nb.predict(X_test)
        # myPredAll = np.append(myPredAll,myPred)
        accuracy_my_all = np.append(accuracy_my_all, accuracy_score(y_test, y_pred_my))
        precision_my_all = np.append(precision_my_all, precision_score(y_test, y_pred_my, zero_division=0))
        recall_my_all = np.append(recall_my_all, recall_score(y_test, y_pred_my, zero_division=0))
        f1_my_all = np.append(f1_my_all, f1_score(y_test, y_pred_my, zero_division=0))


        clf = GaussianNB()
        clf = clf.fit(X_train, y_train)
        y_pred_shelf = clf.predict(X_test)
        # sklPredAll = np.append(sklPredAll, sklPred)
        accuracy_shelf_all = np.append(accuracy_shelf_all, accuracy_score(y_test, y_pred_shelf))
        precision_shelf_all = np.append(precision_shelf_all, precision_score(y_test, y_pred_shelf, zero_division=0))
        recall_shelf_all = np.append(recall_shelf_all, recall_score(y_test, y_pred_shelf, zero_division=0))
        f1_shelf_all = np.append(f1_shelf_all, f1_score(y_test, y_pred_shelf, zero_division=0))

    print('[+] Accuracy')
    print('[shelf] mean {:.3f}; std {:.3f}'.format(accuracy_shelf_all.mean(), accuracy_shelf_all.std()))
    print('[mine]  mean {:.3f}; std {:.3f}'.format(accuracy_my_all.mean(), accuracy_my_all.std()))
    print()

    print('[+] Precision')
    print('[shelf] mean {:.3f}; std {:.3f}'.format(precision_shelf_all.mean(), precision_shelf_all.std()))
    print('[mine]  mean {:.3f}; std {:.3f}'.format(precision_my_all.mean(), precision_my_all.std()))
    print()

    print('[+] Recall')
    print('[shelf] mean {:.3f}; std {:.3f}'.format(recall_shelf_all.mean(), recall_shelf_all.std()))
    print('[mine]  mean {:.3f}; std {:.3f}'.format(recall_my_all.mean(), recall_my_all.std()))
    print()

    print('[+] F1')
    print('[shelf] mean {:.3f}; std {:.3f}'.format(f1_shelf_all.mean(), f1_shelf_all.std()))
    print('[mine]  mean {:.3f}; std {:.3f}'.format(f1_my_all.mean(), f1_my_all.std()))
    print()

if __name__ == "__main__":
    run_naive_bayes()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
