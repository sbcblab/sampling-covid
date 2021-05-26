#import necessary modules
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.naive_bayes import GaussianNB
import matplotlib
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_roc_curve
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
import xgboost as xgb



matplotlib.use('Agg')
# Read in the data with `read_csv()`
sales_data = pd.read_csv("DATA/FLEURY-FM.csv")
sales_teste = pd.read_csv("DATA/FLEURY-FM.csv") # teste



#com tudo
cols = [col for col in sales_data.columns if col in
['Neutrophils#','Eosinophils#',	'Basophils#',	'Lymphocytes#',	'Monocytes#','Neutrophils','Hematocrit','Hemoglobin','Platelets','Meanplateletvolume','RedbloodCells','Lymphocytes','Meancorpuscularhemoglobinconcentration','Leukocytes','Basophils','Meancorpuscularhemoglobin','Eosinophils','Meancorpuscularvolume','Monocytes','Redbloodcelldistributionwidth']]

data = sales_data[cols]
testedata = sales_teste[cols] # teste
target = sales_data['y']
targetteste = sales_teste['y'] #teste


from sklearn.model_selection import train_test_split
#sm = RandomOverSampler(random_state=0)
#data, target = sm.fit_resample(data,target)
X = data.to_numpy()
Y = target.to_numpy()


svmF={}
rfF={}
xgbcF={}
dtF={}
knnF={}
lrF={}
mlpF={}
nbF={}

#svmF[0] = ["cls","run","acc","f1-score","f1-macro","f1-micro","precision","roc-auc-score","recall","balanced-acc","specificity","sensivity","tn","fp","fn","tp"]
#rfF[0] = ["cls","run","acc","f1-score","f1-macro","f1-micro","precision","roc-auc-score","recall","balanced-acc","specificity","sensivity","tn","fp","fn","tp"]
#xgbcF[0] = ["cls","run","acc","f1-score","f1-macro","f1-micro","precision","roc-auc-score","recall","balanced-acc","specificity","sensivity","tn","fp","fn","tp"]
#dtF[0] = ["cls","run","acc","f1-score","f1-macro","f1-micro","precision","roc-auc-score","recall","balanced-acc","specificity","sensivity","tn","fp","fn","tp"]
#knnF[0] = ["cls","run","acc","f1-score","f1-macro","f1-micro","precision","roc-auc-score","recall","balanced-acc","specificity","sensivity","tn","fp","fn","tp"]
#lrF[0] = ["cls","run","acc","f1-score","f1-macro","f1-micro","precision","roc-auc-score","recall","balanced-acc","specificity","sensivity","tn","fp","fn","tp"]



import numpy as np

svmParm = open("ROS-RESULTADOS/svmparm.txt","w")
rfParm = open("ROS-RESULTADOS/rfparm.txt","w")
sgbcParm = open("ROS-RESULTADOS/sgbcparm.txt","w")
dtParm  = open("ROS-RESULTADOS/dtparm.txt","w")
knnParm = open("ROS-RESULTADOS/knnparm.txt","w")
lrParm  = open("ROS-RESULTADOS/lrparm.txt","w")
mlpParm  = open("ROS-RESULTADOS/mlpparm.txt","w")


nbParm  = open("ROS-RESULTADOS/nbparm.txt","w")


#--------


nucleos = 8
runs = 31
from sklearn.model_selection import RandomizedSearchCV
import sys
import os
from sklearn.utils.fixes import loguniform
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses


def NB(run, X_train, X_test, y_train, y_test):
    model = GaussianNB()
    #param_grid = {'alpha': (1, 0.1, 0.01, 0.001, 0.0001, 0.00001)}
    #model = GridSearchCV(GaussianNB(),param_grid,refit=True,verbose=0,n_jobs=nucleos)
    pred=model.fit(X_train, y_train).predict(X_test)
    tn1,fp1,fn1,tp1 = (confusion_matrix(y_test,pred,labels=[0,1]).ravel())
    nbF[run]=[run, round(accuracy_score(y_test,pred),3),
            round(f1_score(y_test,pred),3),
            round(f1_score(y_test,pred,average='macro'),3),
            round(f1_score(y_test,pred,average='micro'),3),
            round(precision_score(y_test,pred),3),
            round(roc_auc_score(y_test,pred),3),
            round(recall_score(y_test,pred),3),
            round(balanced_accuracy_score(y_test,pred),3),
            round(tn1/(tn1+fp1),3),
            round(tp1/(tp1+fn1),3),
            tn1,fp1,fn1,tp1,(tp1/fn1)/(fp1/tn1), (tp1/(tp1+fn1))/(1.0-(tn1/(fp1+tn1))), (1.0 - tp1/(tp1+fn1)) / tn1/(fp1+tn1)]
#    nbParm.write(str(run)+" "+str(model.best_params_)+"\n")


#param_grid = [{'C': [1, 10, 100, 1000],               'kernel': ['linear']},              {'C': [1, 10, 100, 1000],               'gamma': [0.001, 0.0001],               'kernel': ['rbf']}]
#loguniform(1e0, 1e3),

#gama [1,0.1,0.01,0.001,0.0001]

##################
def SVM(run, X_train, X_test, y_train, y_test):
    #model = SVC()
    param_grid = {'C': loguniform(1e0, 1e3), 'gamma': loguniform(1e-4, 1e-3),'kernel': ['rbf','linear']}
    #model = GridSearchCV(SVC(),param_grid,refit=True,verbose=0,n_jobs=nucleos)
    model = RandomizedSearchCV(SVC(),param_grid,refit=True,verbose=0,n_jobs=nucleos)

    pred=model.fit(X_train, y_train).predict(X_test)
    tn1,fp1,fn1,tp1 = (confusion_matrix(y_test,pred,labels=[0,1]).ravel())
    svmF[run]=[run, round(accuracy_score(y_test,pred),3),
            round(f1_score(y_test,pred),3),
            round(f1_score(y_test,pred,average='macro'),3),
            round(f1_score(y_test,pred,average='micro'),3),
            round(precision_score(y_test,pred),3),
            round(roc_auc_score(y_test,pred),3),
            round(recall_score(y_test,pred),3),
            round(balanced_accuracy_score(y_test,pred),3),
            round(tn1/(tn1+fp1),3),
            round(tp1/(tp1+fn1),3),
            tn1,fp1,fn1,tp1,(tp1/fn1)/(fp1/tn1), (tp1/(tp1+fn1))/(1.0-(tn1/(fp1+tn1))), (1.0 - tp1/(tp1+fn1)) / tn1/(fp1+tn1)]
    svmParm.write(str(run)+" "+str(model.best_params_)+"\n")


#'min_samples_split' : [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]}]
##################
def RF(run, X_train, X_test, y_train, y_test):
    param_grid = [{'n_estimators': [50, 100, 200],
                        'criterion': ['gini', 'entropy'],
                        'max_depth': np.arange(3, 10),
                        'min_samples_split' :[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]}]
    #model = GridSearchCV(RandomForestClassifier(),param_grid,refit=True,verbose=0,n_jobs=nucleos)
    model = RandomizedSearchCV(RandomForestClassifier(),param_grid,refit=True,verbose=0,n_jobs=nucleos)
    pred=model.fit(X_train, y_train).predict(X_test)
    tn1,fp1,fn1,tp1 = (confusion_matrix(y_test,pred,labels=[0,1]).ravel())
    rfF[run]=[run, round(accuracy_score(y_test,pred),3),
            round(f1_score(y_test,pred),3),
            round(f1_score(y_test,pred,average='macro'),3),
            round(f1_score(y_test,pred,average='micro'),3),
            round(precision_score(y_test,pred),3),
            round(roc_auc_score(y_test,pred),3),
            round(recall_score(y_test,pred),3),
            round(balanced_accuracy_score(y_test,pred),3),
            round(tn1/(tn1+fp1),3), #Specificity
            round(tp1/(tp1+fn1),3), #Sensivity
            tn1,fp1,fn1,tp1,(tp1/fn1)/(fp1/tn1), (tp1/(tp1+fn1))/(1.0-(tn1/(fp1+tn1))), (1.0 - tp1/(tp1+fn1)) / tn1/(fp1+tn1)]
    rfParm.write(str(run)+" "+str(model.best_params_)+"\n")


# 'learning_rate': [1e-4, 1e-3, 1e-2]}]

##################
def XGBC(run, X_train, X_test, y_train, y_test):
    #model = xgb.XGBClassifier()
    param_grid = [{'n_estimators': [50, 100, 200],
              'max_depth': np.arange(3, 10),
              'learning_rate': loguniform(1e-4, 1e-2)}]
    #model = GridSearchCV(xgb.XGBClassifier(),param_grid,refit=True,verbose=0,n_jobs=nucleos)
    model = RandomizedSearchCV(xgb.XGBClassifier(),param_grid,refit=True,verbose=0,n_jobs=nucleos)

    pred=model.fit(X_train, y_train).predict(X_test)
    tn1,fp1,fn1,tp1 = (confusion_matrix(y_test,pred,labels=[0,1]).ravel())
    xgbcF[run]=[run, round(accuracy_score(y_test,pred),3),
            round(f1_score(y_test,pred),3),
            round(f1_score(y_test,pred,average='macro'),3),
            round(f1_score(y_test,pred,average='micro'),3),
            round(precision_score(y_test,pred),3),
            round(roc_auc_score(y_test,pred),3),
            round(recall_score(y_test,pred),3),
            round(balanced_accuracy_score(y_test,pred),3),
            round(tn1/(tn1+fp1),3), #Specificity
            round(tp1/(tp1+fn1),3), #Sensivity
            tn1,fp1,fn1,tp1,(tp1/fn1)/(fp1/tn1), (tp1/(tp1+fn1))/(1.0-(tn1/(fp1+tn1))), (1.0 - tp1/(tp1+fn1)) / tn1/(fp1+tn1)]

    sgbcParm.write(str(run)+" "+str(model.best_params_)+"\n")

def DT(run, X_train, X_test, y_train, y_test):
    #model = DecisionTreeClassifier()
    param_grid = [{'criterion': ['gini', 'entropy'],
                  'max_depth': np.arange(3, 10),
                  'min_samples_split' : np.arange(0.1, 1.0)}]
    #model = GridSearchCV(DecisionTreeClassifier(),param_grid,refit=True,verbose=0,n_jobs=nucleos)
    model = RandomizedSearchCV(DecisionTreeClassifier(),param_grid,refit=True,verbose=0,n_jobs=nucleos)

    pred=model.fit(X_train, y_train).predict(X_test)
    tn1,fp1,fn1,tp1 = (confusion_matrix(y_test,pred,labels=[0,1]).ravel())
    dtF[run] = [run, round(accuracy_score(y_test,pred),3),
            round(f1_score(y_test,pred),3),
            round(f1_score(y_test,pred,average='macro'),3),
            round(f1_score(y_test,pred,average='micro'),3),
            round(precision_score(y_test,pred),3),
            round(roc_auc_score(y_test,pred),3),
            round(recall_score(y_test,pred),3),
            round(balanced_accuracy_score(y_test,pred),3),
            round(tn1/(tn1+fp1),3), #Specificity
            round(tp1/(tp1+fn1),3), #Sensivity
            tn1,fp1,fn1,tp1,(tp1/fn1)/(fp1/tn1), (tp1/(tp1+fn1))/(1.0-(tn1/(fp1+tn1))), (1.0 - tp1/(tp1+fn1)) / tn1/(fp1+tn1)]

    dtParm.write(str(run)+" "+str(model.best_params_)+"\n")

def KNN(run, X_train, X_test, y_train, y_test):
    #model = KNeighborsClassifier()
    param_grid = [{'n_neighbors': [3, 5, 7, 10, 15, 50],
             'weights': ['uniform', 'distance']}]
    #model = GridSearchCV(KNeighborsClassifier(),param_grid,refit=True,verbose=0,n_jobs=nucleos)

    model = RandomizedSearchCV(KNeighborsClassifier(),param_grid,refit=True,verbose=0,n_jobs=nucleos)

    pred=model.fit(X_train, y_train).predict(X_test)
    tn1,fp1,fn1,tp1 = (confusion_matrix(y_test,pred,labels=[0,1]).ravel())
    knnF[run] = [run, round(accuracy_score(y_test,pred),3),
            round(f1_score(y_test,pred),3),
            round(f1_score(y_test,pred,average='macro'),3),
            round(f1_score(y_test,pred,average='micro'),3),
            round(precision_score(y_test,pred),3),
            round(roc_auc_score(y_test,pred),3),
            round(recall_score(y_test,pred),3),
            round(balanced_accuracy_score(y_test,pred),3),
            round(tn1/(tn1+fp1),3), #Specificity
            round(tp1/(tp1+fn1),3), #Sensivity
            tn1,fp1,fn1,tp1,(tp1/fn1)/(fp1/tn1), (tp1/(tp1+fn1))/(1.0-(tn1/(fp1+tn1))), (1.0 - tp1/(tp1+fn1)) / tn1/(fp1+tn1)]

    knnParm.write(str(run)+" "+str(model.best_params_)+"\n")

def LR(run, X_train, X_test, y_train, y_test):
    model = LogisticRegression()
    pred=model.fit(X_train, y_train).predict(X_test)
    tn1,fp1,fn1,tp1 = (confusion_matrix(y_test,pred,labels=[0,1]).ravel())
    lrF[run] = [run, round(accuracy_score(y_test,pred),3),
            round(f1_score(y_test,pred),3),
            round(f1_score(y_test,pred,average='macro'),3),
            round(f1_score(y_test,pred,average='micro'),3),
            round(precision_score(y_test,pred),3),
            round(roc_auc_score(y_test,pred),3),
            round(recall_score(y_test,pred),3),
            round(balanced_accuracy_score(y_test,pred),3),
            round(tn1/(tn1+fp1),3), #Specificity
            round(tp1/(tp1+fn1),3), #Sensivity
            tn1,fp1,fn1,tp1,(tp1/fn1)/(fp1/tn1), (tp1/(tp1+fn1))/(1.0-(tn1/(fp1+tn1))), (1.0 - tp1/(tp1+fn1)) / tn1/(fp1+tn1)]

#----*---******
def MLP(run, X_train, X_test, y_train, y_test):
    from sklearn.neural_network import MLPClassifier
    param_grid = [{'activation': ['logistic', 'tanh', 'relu'],
             'solver': ['sgd', 'adam'],
             'alpha': [0.0001, 0.001, 0.01],
             'learning_rate_init': [0.0001, 0.001, 0.01],
             'early_stopping': [True, False],
             'batch_size': [16, 64, 128],
             'hidden_layer_sizes': [(10, 10, 2), (5, 10, 5), (10), (10, 20, 5), (10, 10), (100), (30, 10)]}]
    #model = MLPClassifier()
#    model = GridSearchCV(MLPClassifier(),param_grid,refit=True,verbose=0,n_jobs=nucleos)
    model = RandomizedSearchCV(MLPClassifier(),param_grid,refit=True,verbose=0,n_jobs=nucleos)
    pred=model.fit(X_train, y_train).predict(X_test)
    tn1,fp1,fn1,tp1 = (confusion_matrix(y_test,pred,labels=[0,1]).ravel())
    mlpF[run] = [run, round(accuracy_score(y_test,pred),3),
            round(f1_score(y_test,pred),3),
            round(f1_score(y_test,pred,average='macro'),3),
            round(f1_score(y_test,pred,average='micro'),3),
            round(precision_score(y_test,pred),3),
            round(roc_auc_score(y_test,pred),3),
            round(recall_score(y_test,pred),3),
            round(balanced_accuracy_score(y_test,pred),3),
            round(tn1/(tn1+fp1),3), #Specificity
            round(tp1/(tp1+fn1),3), #Sensivity
            tn1,fp1,fn1,tp1,(tp1/fn1)/(fp1/tn1), (tp1/(tp1+fn1))/(1.0-(tn1/(fp1+tn1))), (1.0 - tp1/(tp1+fn1)) / tn1/(fp1+tn1)]
    mlpParm.write(str(run)+" "+str(model.best_params_)+"\n")
#----*---***






for i in tqdm(range(1,runs)):
    X_train, X_test, y_train, y_test = train_test_split(X, Y,stratify=Y,test_size=0.30)

#-------
   # X_train=(X_train-X_train.mean())/X_train.std()
   # X_test=(X_test-X_test.mean())/X_test.std()
    X_train=(X_train-X_train.mean(axis=0))/X_train.std(axis=0)
    X_test=(X_test-X_test.mean(axis=0))/X_test.std(axis=0)

    sm = RandomOverSampler(random_state=0)
    X_trainB,  y_trainB = sm.fit_resample(X_train,y_train)

    NB(i,X_trainB, X_test, y_trainB, y_test)
    print("Running SVM: ",i)
    SVM(i,X_trainB, X_test, y_trainB, y_test)
    print("Running RF: ",i)
    RF(i, X_trainB, X_test, y_trainB, y_test)
    print("Running XGBC: ",i)
    XGBC(i, X_trainB, X_test, y_trainB, y_test)
    print("Running DT: ",i)
    DT(i, X_trainB, X_test, y_trainB, y_test)
    print("Running KNN: ",i)
    KNN(i, X_trainB, X_test, y_trainB, y_test)
    print("Running LR: ",i)
    LR(i, X_trainB, X_test, y_trainB, y_test)
    print("Running MLP: ",i)
    MLP(i, X_trainB, X_test, y_trainB, y_test)


def saveCSV(dic,nome):
    from numpy import asarray
    from numpy import savetxt
    suporte = []
    for i in dic.keys():
        suporte.append(dic[i])
    # define data
    data = asarray(suporte)
    # save to csv file
    savetxt("ROS-RESULTADOS/"+nome+".csv", data, delimiter=',')


saveCSV(svmF,"SVM-ROS")
saveCSV(rfF,"RF-ROS")
saveCSV(xgbcF,"XGBC-ROS")
saveCSV(dtF,"DT-ROS")
saveCSV(knnF,"KNN-ROS")
saveCSV(lrF,"LR-ROS")
saveCSV(mlpF,"MLP-ROS")
saveCSV(nbF,"NB-ROS")

svmParm.close()
rfParm.close()
sgbcParm.close()
dtParm.close()
knnParm.close()
lrParm.close()
mlpParm.close()
nbParm.close()
