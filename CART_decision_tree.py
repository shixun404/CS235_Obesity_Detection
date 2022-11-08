import pandas as pd
def preprocessing() -> pd.DataFrame:
    df = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')

    df['Label'] = df['NObeyesdad'].str.startswith('Obesity').astype(int)
    del df['NObeyesdad']
    del df['Height']
    del df['Weight']

    freq_map = {'no': 0, 'Sometimes': 1,
    'Frequently': 2, 'Always': 3}

    yes_no_map = {'yes': 1, 'no': 0}

    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
    df['family_history_with_overweight'] = df['family_history_with_overweight'].map(yes_no_map)
    df['FAVC'] = df['FAVC'].map(yes_no_map)
    df['CAEC'] = df['CAEC'].map(freq_map)
    df['SMOKE'] = df['SMOKE'].map(yes_no_map)
    df['SCC'] = df['SCC'].map(yes_no_map)
    df['CALC'] = df['CALC'].map(freq_map)
    df['MTRANS'] = df['MTRANS'].map({'Automobile': 0, 'Motorbike': 0, 'Public_Transportation': 0, 'Walking': 1, 'Bike': 1})

    normalized_df = (df-df.min())/(df.max()-df.min())
    df = normalized_df
    return df


from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
df = preprocessing()
X = df.drop(columns=['Label'])
y = df['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

# X = [[0, 0], [1, 1]]
# Y = [0, 1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
print(accuracy_score(y_test, clf.predict(X_test)))
print(precision_score(y_test, clf.predict(X_test)))
print(recall_score(y_test, clf.predict(X_test)))
print(f1_score(y_test, clf.predict(X_test)))
print(confusion_matrix(y_test, clf.predict(X_test)))