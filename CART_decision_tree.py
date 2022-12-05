import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

def greedy_split_gini(x,y):
    # condition dictionary
    gini = 0
    df = x.join(y)
    min_gini = 100
    label_value = None
    for label in x.columns:
        label_stat = df[label].value_counts()
        s = label_stat.sum()
        for key in label_stat.index:
            tmp = ((df[df[label] <= key])['Label']).value_counts()
            if tmp.size > 0:
                value = tmp.sum()
                p = tmp.values[0] / tmp.sum()
                gini = value / s * (1 - p ** 2 - (1-p) ** 2)
            tmp = ((df[df[label] > key])['Label']).value_counts()
            if tmp.size > 0:
                p = tmp.values[0] / tmp.sum()
                value = tmp.sum()
                gini += value / s * (1 - p ** 2 - (1-p) ** 2)
        if gini < min_gini:
            label_ = label
            label_value = key
            min_gini = gini
    return label_, label_value, min_gini
a_1 = 0
a_2 = 0
# for key in predict.index:
#         if key == 0.0:
#             a_1 = predict.loc[key]
#         else:
#             a_2 = predict.loc[key]
#     if a_1 > a_2:
#         predict = False
#     else:
#         predict = True
class Node:
    def __init__(self,):
        self.child = []
        self.label = None
        self.predict = None
        self.gini = 0
        self.label_value = None

def build_dfs(x, y):
    node = Node()
    min_gini = 100
    label_ = None
    tmp = y.value_counts()
    predict_no_child = False
    a_1 = 0
    a_2 = 0
    for key in tmp.index:
        if key == 0.0:
            a_1 = tmp.loc[key]
        else:
            a_2 = tmp.loc[key]
    if a_1 > a_2:
        predict_no_child = False
    else:
        predict_no_child = True
    p = tmp.values[0] / tmp.sum()
    gini_no_child = (1 - p ** 2 - (1-p) ** 2)
    label_, label_value, min_gini = greedy_split_gini(x,y)
    if gini_no_child <= min_gini:
        node.label = label_
        node.predict = predict_no_child
        node.gini = gini_no_child
        node.label_value = label_value
        return node
    else:
        node.label = label_
        node.predict = None
        node.gini = min_gini
        node.label_value = label_value
        index = x.index[x[label_] <= label_value]
        child_node = build_dfs(x.loc[index], y.loc[index])
        node.child.append(child_node)
        index = x.index[x[label_] > label_value]
        child_node = build_dfs(x.loc[index], y.loc[index])
        node.child.append(child_node)
    return node

def preprocessing() -> pd.DataFrame:
    df = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')
    # df = df.loc[:10 * b]
    df['Label'] = df['NObeyesdad'].str.startswith('Obesity').astype(int)
    del df['NObeyesdad']
    del df['Height']
    del df['Weight']

    freq_map = {'no': 0, 'Sometimes': 1,
    'Frequently': 2, 'Always': 3}

    yes_no_map = {'yes': 1, 'no': 0}
    df['Age'] = (df['Age']).round()
    df['CH2O'] = (df['CH2O'] * 10).round() / 10
    df['FAF'] = (df['FAF'] * 10).round() / 10
    df['TUE'] = (df['TUE'] * 10).round() / 10
    df['FCVC'] = df['FCVC'].round()
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
    df['family_history_with_overweight'] = df['family_history_with_overweight'].map(yes_no_map)
    df['FAVC'] = df['FAVC'].map(yes_no_map)
    df['CAEC'] = df['CAEC'].map(freq_map)
    df['SMOKE'] = df['SMOKE'].map(yes_no_map)
    df['SCC'] = df['SCC'].map(yes_no_map)
    df['CALC'] = df['CALC'].map(freq_map)
    df['MTRANS'] = df['MTRANS'].map({'Automobile': 0, 'Motorbike': 0, 'Public_Transportation': 0, 'Walking': 1, 'Bike': 1})
    normalized_df = (df-df.min())/(df.max()-df.min() + 1e-7)
    df = normalized_df
    return df

df = preprocessing()
X = df.drop(columns=['Label'])
y = df['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

node = build_dfs(X_train, y_train)

k =0

cnt = 0
for i in range(X_test.shape[0]):
    node_ = node
    while (len(node_.child) > 0):
        # print(node_.label, len(node_.child))
        k+=1
        if len(node_.child) > 0:
            # print(j,node_.label, node_.child[j].label_value, (X.iloc[i])[node_.label])
            if (X_test.iloc[i])[node_.label] <= node_.label_value:
                node_ = node_.child[0]
            else:
                node_ = node_.child[1]
            # print(j, node_.child[j].label_value, (X_test.iloc[i])[node_.label])
            
    print(f"{i}-th row is obesity: {node_.predict}, {y_test.iloc[i]}")
    if node_.predict == False and y_test.iloc[i] == 0.0:
        cnt += 1
    if node_.predict == True and y_test.iloc[i] != 0.0:
        cnt += 1
print(cnt / y_test.shape[0])
    

assert 0
# print(X.join(y))
print(y.value_counts().sum())

assert 0

for i in X_train.columns:
    print(i, X_train[i].unique().shape)
# # X = [[0, 0], [1, 1]]
# # Y = [0, 1]
# clf = tree.DecisionTreeClassifier()
# clf = clf.fit(X_train, y_train)
# print(accuracy_score(y_test, clf.predict(X_test)))
# print(precision_score(y_test, clf.predict(X_test)))
# print(recall_score(y_test, clf.predict(X_test)))
# print(f1_score(y_test, clf.predict(X_test)))
# print(confusion_matrix(y_test, clf.predict(X_test)))