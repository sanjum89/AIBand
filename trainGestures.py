#Import Statements
#---------------------------------------------------
from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np
from sklearn.externals import joblib
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
import glob
import os
#---------------------------------------------------


os.system('cls')
print("Initializing")
print()


#Directory Lists
#---------------------------------------------------
diradr = os.getcwd()
curPath = diradr.replace(os.sep, '\\')

main_dir = curPath+'\\Data'
single_dir = curPath+'\\Data\\dataSetSingle'
binary_dir = curPath+'\\Data\\dataSetBinary'
multi_dir = curPath+'\\Data\\dataSetMulti'
#---------------------------------------------------


#Joblib File Names
#---------------------------------------------------
joblib_gesTrain = "gesTrain.elxr"
joblib_detType = "detType.elxr"
#---------------------------------------------------


#Variables
#---------------------------------------------------
clf=SVC()
X=0
y=0
scaler=0
X_scaled=0
clfr=0
X_train=0
X_test=0
y_train=0
y_test=0
clfr=0
filename = ''
#---------------------------------------------------


#Select Dataset File
#---------------------------------------------------
def fileSelection(clfType):
    global dataFile
    temp_list=[]
    nFiles=0
    flistMsg =""
    if(clfType=='b'):
        path = binary_dir+'\\*.csv'
        for fname in glob.glob(path):
            temp_list.append(fname)
        nFiles = len(temp_list)
        print("Files Found : ",nFiles)
        for tl in range(0,nFiles):
            ftemp=str(''.join(filter(str.isdigit,temp_list[tl])))
            flistMsg+=str(tl)+" - dataset"+str(ftemp)+", "

        flistMsg=flistMsg[:-2]
        flistMsg = "Select File : "+flistMsg+" : "
        selFile = int(input(flistMsg))
        temfilename = temp_list[selFile]
        f1=str(''.join(filter(str.isdigit,temfilename)))
        dataFile = binary_dir+"\\dataset"+str(f1)+".csv"
        selectf=str(''.join(filter(str.isdigit,dataFile)))
        print("DataFile Selected : dataSet"+str(selectf))
        joblib.dump(selectf,joblib_gesTrain)
    else:
        dataFile = multi_dir+"\\datasetComplete.csv"
        print("DataFile Selected : ",dataFile)
#---------------------------------------------------


#Select Preprocessor for Scaling
#---------------------------------------------------
def preproc(prechoice):
    global scaler
    global X_scaled
    #PreProcessing. Mean = 0, variance = 1
    if(prechoice==1):
        scaler = preprocessing.StandardScaler().fit(X)
        X_scaled = scaler.transform(X)
        print("Scaler :",scaler)
    #PreProcessing. Normalize Each feature
    if(prechoice==2):
        scaler = preprocessing.Normalizer().fit(X)
        X_scaled = scaler.transform(X)
        print("Scaler :",scaler)
#---------------------------------------------------


#Load dataset from file to np array
#---------------------------------------------------
def getData():
    global X
    global y

    #Training Data File
    data = np.loadtxt(dataFile, delimiter=",")
    print("Data Located. Rows : ",len(data))


    #Get Data from csv to numpy array
    X = np.asarray(data[:, 0:406])
    y = np.asarray(data[:,406])
    
    print("Data Load Complete")
    print("---------------------------")
    print("Dataset  :",np.size(X,0))
    print("Features :",np.size(X,1))
    print("Gestures :",(len(np.unique(y))))
    unique, counts = np.unique(y,return_counts=True)
    uq=np.asarray((unique, counts)).T
    print("Gestures and Count - ")
    print(uq)
    print("---------------------------")
#---------------------------------------------------


#Split Data
#---------------------------------------------------
def tsplit():
    global X_train
    global X_test
    global y_train
    global y_test
    global clf
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=0)
#---------------------------------------------------


#Fit Classifier on given data
#---------------------------------------------------
def clffit():
    global clf
    clf = clf.fit(X_train, y_train)
    print("Classifier : ",clf)
    print("Score : ",clf.score(X_test,y_test))
#---------------------------------------------------


#Save Trained Model
#---------------------------------------------------
def model_save():
    joblib_modelfile = "joblib_model.pkl"
    joblib_scalerfile = "joblib_scaler.pkl"
    joblib.dump(clf, joblib_modelfile)
    joblib.dump(scaler, joblib_scalerfile)
    print("Classifier being saved")
    print(clf)
    print("Model Saved")
    print("")
#---------------------------------------------------


# Function to sequentially execute Scale, Train and Save Model functions.
#---------------------------------------------------
def training():
    tsplit()
    print()
    print("#########################################")
    clffit()
    print("#########################################")
    print()
    model_save()
    print("#########################################")
    print()
#---------------------------------------------------


#Various Classifiers
#---------------------------------------------------
def svcBin():
    global clf
    clf = SVC(kernel='rbf',gamma=100, C=1, class_weight='balanced', decision_function_shape='ovr')
    training()

def svcMulti():
    global clf
    clf = SVC(kernel='rbf',gamma=.001, C=1, class_weight='balanced', decision_function_shape='ovo')
    training()

def lrBin():
    global clf
    clf = LogisticRegression(solver='lbfgs', class_weight='balanced', C=1, multi_class='ovr', n_jobs=-1)
    training()

def lrMulti():
    global clf
    clf = LogisticRegression(solver='lbfgs', class_weight='balanced', C=1, multi_class='multinomial', n_jobs=-1)
    training()

def getknn():
    global clf
    clf = KNeighborsClassifier(n_neighbors=15)

    #Save Model and Scaler for prediction
    joblib_modelfile = "joblib_model.pkl"
    joblib_scalerfile = "joblib_scaler.pkl"
    joblib.dump(clf, joblib_modelfile)
    joblib.dump(scaler, joblib_scalerfile)
    print(clf)
    print("Model Saved")
    print("")

def getLrCV():
    global clf
    print("Starting LRCV")
    clf = LogisticRegressionCV(cv=5, n_jobs=-1,class_weight='balanced', random_state=0, multi_class='multinomial', scoring='accuracy')

    #Save Model and Scaler for prediction
    joblib_modelfile = "joblib_model.pkl"
    joblib_scalerfile = "joblib_scaler.pkl"
    joblib.dump(clf, joblib_modelfile)
    joblib.dump(scaler, joblib_scalerfile)
    print(clf)
    print("Model Saved")
    print("")
#---------------------------------------------------


# Main Program
#---------------------------------------------------
while(True):

    global choice
    global prechoice

    gscv = str(input("Load GSCV Data y/n ? "))
    
    if(gscv=='y'):
        dsClf = joblib.load('dsClf.elxr')
        dsScaler = joblib.load('dsScaler.elxr')
        paramDict = joblib.load('paramDict.elxr')
        print("Classifier Loaded : ",dsClf)
        print("Scaler Loaded     : ",dsScaler)
        print("Parameters Loaded : ",paramDict)

        if (dsClf=='SVC'):
            choice=1
        elif (dsClf=='SVCMulti'):
            choice=2
        elif (dsClf=='LR'):
            choice=3
        elif (dsClf=='LRMulti'):
            choice=4
        else:
            print("No Defined GSCV Classifier")

        if (dsScaler=='MeVar'):
            prechoice=1
        elif (dsScaler=='Norm'):
            prechoice=2
        else:
            print("No Defined GSCV Scaler")
        
    else:
        print("Choose wisely")
        choice = int(input("Classifier ? 1 - SVC Binary, 2 - SVC Multi, 3 - LR Binary, 4 - LR Multi : "))
        print()
        prechoice = int(input("PreProcessor ? 1 - MeVar, 2 - Norm : "))

    if choice==1:
        fileSelection('b')
        getData()
        preproc(prechoice)
        svcBin()
        joblib.dump('b',joblib_detType)
    elif choice==2:
        fileSelection('m')
        getData()
        preproc(prechoice)
        svcMulti()
        joblib.dump('m',joblib_detType)
    elif choice==3:
        fileSelection('b')
        getData()
        preproc(prechoice)
        lrBin()
        joblib.dump('b',joblib_detType)
    elif choice==4:
        fileSelection('m')
        getData()
        preproc(prechoice)
        lrMulti()
        joblib.dump('m',joblib_detType)
    else:
        print("Wrong Input")
#---------------------------------------------------