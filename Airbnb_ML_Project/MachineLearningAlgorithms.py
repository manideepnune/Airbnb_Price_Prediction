import matplotlib.pyplot as plt
import pandas
import pandas as pd
import seaborn as sns
from sklearn import linear_model
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score
from sklearn.metrics import max_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from rulefit import RuleFit


def read_and_split_data(input,output):
    input_train, input_test, output_train, output_test = train_test_split(input, output, test_size=0.1, random_state=0)
    return (input_train, input_test, output_train, output_test)

def plotInputs(input_train, input_test, output_train,output_test):
    l = []
    for i in input_train.values:
        l.append(sum(i))

    sns.regplot(l, output_train)
    plt.suptitle("Airbnb price prediction(Training Set)")
    plt.show()

    l = []
    for i in input_test.values:
        l.append(sum(i))
    sns.regplot(l, output_test)
    plt.suptitle("Airbnb price prediction(Testing Set)")
    plt.show()


def plotOutputs(mlobj , input_train, input_test, output_train,output_test ,title):
    l = []
    for i in input_test.values:
        l.append(sum(i))

    sns.regplot(l, mlobj.predict(input_test))
    plt.suptitle("Airbnb price prediction for algorithm  :  "+title)
    plt.show()



def kNNRegression(input_train, input_test, output_train):
    neigh = KNeighborsRegressor(n_neighbors=2)
    neigh.fit(input_train, output_train)
    print("Algorithm score : "+str(neigh.score(input_train, output_train)))
    plotOutputs(neigh, input_train, input_test, output_train,output_test ,"KNeighborsRegressor")
    return neigh.predict(input_test)

def NaiveRegression(input_train, input_test, output_train):
    clf = linear_model.BayesianRidge()
    clf.fit(input_train, output_train)
    print("Algorithm score : " + str(clf.score(input_train, output_train)))
    plotOutputs(clf, input_train, input_test, output_train, output_test, "BayesianRidge")
    return clf.predict(input_test)

def decisionTreeRegression(input_train, input_test, output_train):
    clf = DecisionTreeRegressor(random_state=0)
    clf.fit(input_train, output_train)
    print("Algorithm score : " + str(clf.score(input_train, output_train)))
    plotOutputs(clf, input_train, input_test, output_train, output_test, "DecisionTreeRegressor")
    return clf.predict(input_test)

# def ruleBasedRegression_2(input_train, input_test, output_train):
#     clf = RuleFit()
#     clf.fit(input_train.as_matrix(), output_train)
#     #print("Algorithm score : " + str(clf.score(input_train, output_train)))
#     #plotOutputs(clf, input_train, input_test, output_train, output_test, "ruleBasedRegression")
#     return clf.predict(input_test.as_matrix())

def ruleBasedRegression(input_train, input_test, output_train):
    clf = DummyRegressor()
    clf.fit(input_train, output_train)
    print("Algorithm score : " + str(clf.score(input_train, output_train)))
    plotOutputs(clf, input_train, input_test, output_train, output_test, "ruleBasedRegression")
    return clf.predict(input_test)

def RandomForestRegression(input_train, input_test, output_train):
    clf = RandomForestRegressor()
    clf.fit(input_train, output_train)
    print("Algorithm score : " + str(clf.score(input_train, output_train)))
    plotOutputs(clf, input_train, input_test, output_train, output_test, "RandomForestRegression")
    return clf.predict(input_test)

def votingEnsembler(input_train, input_test, output_train):
    r1 = LinearRegression()
    r2 = RandomForestRegressor(n_estimators=10, random_state=1)
    er = VotingRegressor([('lr', r1), ('rf', r2)])
    er.fit(input_train, output_train)
    print("Algorithm score : " + str(er.score(input_train, output_train)))
    plotOutputs(er, input_train, input_test, output_train, output_test, "VotingRegressor")
    return er.predict(input_test)



def variance_score(output_test,output_predict):
    return explained_variance_score(output_test,output_predict)
def max_error_score(output_test,output_predict):
    return max_error(output_test,output_predict)
def mean_ae(output_test,output_predict):
    return mean_absolute_error(output_test,output_predict)
def mse(output_test,output_predict):
    return mean_squared_error(output_test,output_predict)

def median_ae(output_test,output_predict):
    return median_absolute_error(output_test,output_predict)
def r_square(output_test,output_predict):
    return r2_score(output_test,output_predict)
def rmse(output_test,output_predict):
    return mse(output_test,output_predict)**0.5
def regressionEvaluation(output_test,output_predict):

    print("variance_score")
    print(variance_score(output_test,output_predict))

    print("max_error_score")
    print(max_error_score(output_test, output_predict))

    print("mean_ae")
    print(mean_ae(output_test, output_predict))

    print("mse")
    print(mse(output_test, output_predict))


    print("median_ae")
    print(median_ae(output_test, output_predict))

    print("r_square")
    print(r_square(output_test, output_predict))

    print("RMSE")
    print(rmse(output_test,output_predict))



def regressionAlgo(input_train, input_test, output_train, output_test):

    plotInputs(input_train, input_test, output_train, output_test)

    print("-----------------------------------------------")
    print("K-NN Regression")
    print("-----------------------------------------------")
    output_pred = kNNRegression(input_train, input_test, output_train)
    print(output_pred)
    regressionEvaluation(output_test,output_pred)

    print("-----------------------------------------------")
    print("Naive Regression")
    print("-----------------------------------------------")
    output_pred = NaiveRegression(input_train, input_test, output_train)
    print(output_pred)
    regressionEvaluation(output_test, output_pred)

    print("-----------------------------------------------")
    print("decision tree Regression")
    print("-----------------------------------------------")
    output_pred = decisionTreeRegression(input_train, input_test, output_train)
    print(output_pred)
    regressionEvaluation(output_test, output_pred)

    print("-----------------------------------------------")
    print("rule Regression")
    print("-----------------------------------------------")
    output_pred = ruleBasedRegression(input_train, input_test, output_train)
    print(output_pred)
    regressionEvaluation(output_test, output_pred)

    # print("-----------------------------------------------")
    # print("rule fit")
    # print("-----------------------------------------------")
    # output_pred = ruleBasedRegression(input_train, input_test, output_train)
    # print(output_pred)
    # regressionEvaluation(output_test, output_pred)

    print("-----------------------------------------------")
    print("Random Forest Regression")
    print("-----------------------------------------------")
    output_pred = RandomForestRegression(input_train, input_test, output_train)
    print(output_pred)
    regressionEvaluation(output_test, output_pred)

    print("-----------------------------------------------")
    print("Voting Ensembler ")
    print("-----------------------------------------------")
    output_pred = votingEnsembler(input_train, input_test, output_train)
    print(output_pred)
    regressionEvaluation(output_test, output_pred)

data = pandas.read_csv('./listings_preprocessed.csv')
output = data['price']
input = data.drop(['price'], axis=1)


class_dist = data.groupby('beds').size()
class_label = pd.DataFrame(class_dist, columns=['Size'])
plt.figure(figsize=(10, 10))
sns.barplot(x=class_label.index, y='Size', data=class_label)
plt.show()

class_dist = data.groupby('cat(s)').size()
class_label = pd.DataFrame(class_dist, columns=['Size'])
plt.figure(figsize=(10, 10))
sns.barplot(x=class_label.index, y='Size', data=class_label)
plt.show()


class_dist = data.groupby('bathrooms').size()
class_label = pd.DataFrame(class_dist, columns=['Size'])
plt.figure(figsize=(10, 10))
sns.barplot(x=class_label.index, y='Size', data=class_label)
plt.show()

class_dist = data.groupby('accommodates').size()
class_label = pd.DataFrame(class_dist, columns=['Size'])
plt.figure(figsize=(10, 10))
sns.barplot(x=class_label.index, y='Size', data=class_label)
plt.show()



print("-----------------------------------------------")
print("RAW DATA")
print("-----------------------------------------------")
input_train, input_test, output_train, output_test = read_and_split_data(input , output)
regressionAlgo(input_train, input_test, output_train, output_test)

print("-----------------------------------------------")
print("Variance Threshold feature selection")
print("-----------------------------------------------")
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
new_input = sel.fit_transform(input)
input_train, input_test, output_train, output_test = read_and_split_data(pd.DataFrame(new_input) , output)
regressionAlgo(input_train, input_test, output_train, output_test)

