##########################  Rain Prediction Project  ##########################
#Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import jaccard_score, f1_score, log_loss
from sklearn.metrics import accuracy_score, confusion_matrix
import itertools

###Import Data###
filepath = 'weather.csv'
df = pd.read_csv(filepath)

###Data Pre-Processing###
#Clean Data
df.drop('Unnamed: 0', axis=1, inplace=True)
#Convert categorical values to Numeric dummys
df_processed = pd.get_dummies(data=df, columns=['RainToday', 'WindGustDir', 
                                                'WindDir9am', 'WindDir3pm'])
df_processed['RainTomorrow'].replace(['No', 'Yes'], [0,1], inplace=True)
#Drop Date colunn
df_processed.drop('Date', axis=1, inplace=True)
#Convert variables to floats
df_processed = df_processed.astype('float')
#Set X and y
features = df_processed.drop('RainTomorrow',axis=1)
y = df_processed['RainTomorrow']


### Multiple Linear Regression Model ###
def MLR():
    #Train Test Split
    x_train,x_test,y_train,y_test = train_test_split(features, y, test_size=0.2,
                                                     random_state=10)
    print(y_test.shape)
    #Fit and Predict Model
    lm = LinearRegression()
    lm.fit(x_train, y_train)
    yhat = lm.predict(x_test)
    #Calculate Errors
    mae_lm = mean_absolute_error(y_test, yhat)
    msqe_lm = mean_squared_error(y_test, yhat)
    rsq_lm = r2_score(y_test,yhat)
    frame = {'MAE':[mae_lm],
             'MSE':msqe_lm,
             'R2':rsq_lm}
    report_mlr = pd.DataFrame(frame)
    return report_mlr

MLR()
### K-Nearest Neighbors Model ###
def KNN():
    #Set X and y
    X_knn = features.values
    X_knn = StandardScaler().fit(X_knn).transform(X_knn.astype(float))
    y_knn = df_processed['RainTomorrow'].values
    #Train Test Split
    x_train,x_test,y_train,y_test = train_test_split(X_knn, y_knn, test_size=0.2,
                                                     random_state=10)
    numbs = np.arange(1,16)
    ack_list = list()
    jck_list = list()
    fsk_list = list()
    for i in numbs:
        #Fit KNN Model
        k = i
        KNN = KNeighborsClassifier(n_neighbors=k).fit(x_train, y_train)
        #Predict KNN Model
        yhat = KNN.predict(x_test)
        #Calculate Metrics
        acc_score_knn = accuracy_score(y_test, yhat)
        ack_list.append(acc_score_knn)
        jac_score_knn = jaccard_score(y_test, yhat, pos_label=0)
        jck_list.append(jac_score_knn)
        f1_score_knn = f1_score(y_test, yhat, average='weighted')
        fsk_list.append(f1_score_knn)
    print(fsk_list)


### Decision Tree Model ###
def decision_tree():
    #Set X and y
    X = features.values
    y = df_processed['RainTomorrow'].values
    #Train Test Split
    x_train,x_test,y_train,y_test = train_test_split(X, y, test_size=0.2,
                                                     random_state=10)
    #Fit Decision Tree Model
    dTree = DecisionTreeClassifier(criterion='entropy')
    dTree.fit(x_train, y_train)
    #Predict Decision Tree Model
    yhat_tree = dTree.predict(x_test)
    #Calculate Metrics
    acc_score_tree = accuracy_score(y_test, yhat_tree)
    jac_score_tree = jaccard_score(y_test, yhat_tree, pos_label=0)
    f1_score_tree = f1_score(y_test, yhat_tree)
    print(acc_score_tree,
          jac_score_tree, 
          f1_score_tree)
    

### Logistic Regression Model ###
def log_gegression():
    #Set X and y
    df_processed['RainTomorrow'] = df_processed['RainTomorrow'].astype('int')
    y = df_processed['RainTomorrow'].values
    X = features.values
    X = StandardScaler().fit(X).transform(X)
    #Train Test Split
    x_train,x_test,y_train,y_test = train_test_split(X, y, test_size=0.2, 
                                                     random_state=10)
    #Fit Log Regression Model
    log_reg = LogisticRegression(solver='liblinear')
    log_reg.fit(x_train, y_train)
    #Predict Log Regression Model
    yhat = log_reg.predict(x_test)
    yhat_prob = log_reg.predict_proba(x_test)
    #Calculate Metrics
    acc_score_log = accuracy_score(y_test, yhat)
    jac_score_log = jaccard_score(y_test, yhat, pos_label=0)
    f1_score_log = f1_score(y_test, yhat)
    log_loss_log = log_loss(y_test, yhat_prob)
    cnf_matrix_log = confusion_matrix(y_test, yhat, labels=[0,1])
    print(acc_score_log, 
          jac_score_log, 
          f1_score_log, 
          log_loss_log,
          cnf_matrix_log)
    
### Support Vector Machine (SVM) Model ###
def support_vector_machine():
    #Set X and y
    X = features.values
    y = df_processed['RainTomorrow'].values
    #Train Test Split
    x_train,x_test,y_train,y_test = train_test_split(X, y, test_size=0.2,
                                                     random_state=10)
    #Fit SVM Model
    svm1 = svm.SVC(kernel='linear')
    svm1.fit(x_train, y_train)
    #Predict SVM Model
    yhat = svm1.predict(x_test)
    #Calculate Metrics
    acc_score_svm = accuracy_score(y_test, yhat)
    jac_score_svm = jaccard_score(y_test, yhat, pos_label=0)
    f1_score_svm = f1_score(y_test, yhat)
    cnf_matrix_svm = confusion_matrix(y_test, yhat, labels=[0,1])
    print(acc_score_svm, 
          jac_score_svm, 
          f1_score_svm, 
          cnf_matrix_svm)
    return cnf_matrix_svm

#Confusion Matrix Plot Function
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



#Gather Data from Functions
#report_mlr = MLR()

#cnf_matrix_svm = support_vector_machine()
#Plotting Confusion Matrix (SVM)
'''plt.figure
plot_confusion_matrix(cnf_matrix_svm,classes=['No Rain(0)','Yes Rain(1)'],
                      normalize=False, title='Confusion Matrix for SVM')'''




#Save Data
#df.to_csv('weather.csv')
