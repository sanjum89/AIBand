import serial
from datetime import datetime
import numpy as np
import csv
import sys
import click
import os
from os import walk
import warnings
import glob
import pandas as pd
from sklearn.externals import joblib
import time

#For GridSearchCV
#---------------------------------------------------
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn import datasets
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
#---------------------------------------------------

os.system('cls')
print("Initializing")

#Directory Lists
#---------------------------------------------------
diradr = os.getcwd()
curPath = diradr.replace(os.sep, '\\')

main_dir = curPath+'\\Data'
single_dir = curPath+'\\Data\\dataSetSingle'
binary_dir = curPath+'\\Data\\dataSetBinary'
multi_dir = curPath+'\\Data\\dataSetMulti'
#---------------------------------------------------

joblib_gesDict = "gestDict.elxr" #Gesture Dictionary
joblib_message = "message.elxr" #Message for User Input, with Gesture and Assigned Number
joblib_gestNum = "gesturenumber.elxr" # Total Number of Gestures
joblib_clfType = "clfType.elxr" #Binary or Multi

joblib_dsClf = "dsClf.elxr" #GSCV Classifier Selected
joblib_dsScaler = "dsScaler.elxr" #GSCV Scaler Selected
joblib_paramDict = "paramDict.elxr" #GSCV Tuner Parameter Dictionary

filename_base  = 'dataset'
extension = 'csv'
filename = ''

enable=1

#Custom Print Job
def newprint(pridat):
    if(enable==1):
        print(pridat)


def csvconcat():
    print("Starting Concat. Please Wait")
    print("------------------------------")
    fname_list=[]
    path = single_dir+'\\*.csv'
    for fname in glob.glob(path):
        fname_list.append(fname)

    fcomb_list = ['']*2
    dfList = []

    for filename in fname_list:
        df=pd.read_csv(filename,header=None)
        dfList.append(df)
    concatDf=pd.concat(dfList,axis=0)
    compFile_path=multi_dir+"\\datasetComplete.csv"
    concatDf.to_csv(compFile_path, index=False, header=None)

    concatDf.iloc[0:0]
    dfList.clear()

    fileCount = len(fname_list)

    for x in range (0, fileCount):
        fcomb_list [0] = fname_list[x]
        s1=str(''.join(filter(str.isdigit, fcomb_list[0])))
        
        for y in range (x+1, fileCount):
            fcomb_list [1] = fname_list[y]
            s2=str(''.join(filter(str.isdigit, fcomb_list[1])))
            print("Gesture Files : ",s1,", ",s2,end='' )
            print()
            for filename in fcomb_list:
                binfileName=filename_base+s1+s2+'.'+extension
                # print(filename, end='')
                # print(" ", end='')
                df=pd.read_csv(filename,header=None)
                dfList.append(df)
            concatDf=pd.concat(dfList,axis=0)
            binaryFile_path = binary_dir+"\\"+binfileName
            concatDf.to_csv(binaryFile_path, index=False, header=None)
            concatDf.iloc[0:0]
            dfList.clear()
    print("------------------------------")
    print("#############################################################################")
    print()


#Preprocessor for Data
def preproc(prechoice):
    global scaler
    global X_scaled
    #Mean = 0, variance = 1
    if(prechoice==1):
        scaler = preprocessing.StandardScaler().fit(X)
        X_scaled = scaler.transform(X)
        print("Scaler :",scaler)
    #Normalize Each feature
    if(prechoice==2):
        scaler = preprocessing.Normalizer().fit(X)
        X_scaled = scaler.transform(X)
        print("Scaler :",scaler)
    if(prechoice==3):
        return
#GetData from
def getData():
    global X,y
    global X_train, X_test, y_train, y_test
    global selectScaler

    data = np.loadtxt(dataFile, delimiter=",")
    print("Data Located. Rows : ",len(data))

    #Get Data from csv to numpy array
    X = np.asarray(data[:, 0:406])
    y = np.asarray(data[:,406])
    print("Data Load Complete")
    print("-------------------------------------------------------------------")
    print("Dataset  :",np.size(X,0))
    print("Features :",np.size(X,1))
    print("Gestures :",(len(np.unique(y))))
    unique, counts = np.unique(y,return_counts=True)
    uq=np.asarray((unique, counts)).T
    print("Gestures and Count - ")
    print(uq)
    print("-------------------------------------------------------------------")
    print()
    print("Data")
    print(X)
    print()
    print("Select Your PreProcessor")
    prechoice = int(input("1 - MeVar, 2 - Norm, 3-NoPreProc : "))
    
    if (prechoice==1):
        selectScaler = 'MeVar'
    else:
        selectScaler = 'Norm'
    
    preproc(prechoice)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=0)
    print("")
    print("Scaled Data")
    print(X_scaled)
    print("")
    print("Ready for GridSearchCV")
    print("")

def svcbinPar():
    global tuned_parameters
    global classifier
    gamma_range = [ 1e-4,1e-3,1e-2, 1, 1e2]
    C_range = [1e-1, 1, 1e1, 1e2, 1e3]
    coef0_range = [0.0,0.1]
    degree_range = [0,1,2,3,4,5,6]

    tuned_parameters = [{'kernel':['rbf','linear'], 'gamma':gamma_range, 'C':C_range, 'class_weight':['balanced']},
        {'kernel':['sigmoid'], 'gamma':gamma_range, 'C':C_range, 'coef0':coef0_range, 'class_weight':['balanced']},
            {'kernel':['poly'], 'gamma':gamma_range, 'C':C_range, 'coef0':coef0_range, 'class_weight':['balanced'], 'degree':degree_range}]

    classifier = SVC(decision_function_shape='ovr')

def svcPar():
    global tuned_parameters
    global classifier
    gamma_range = [ 1e-4,1e-3,1e-2, 1, 1e2]
    C_range = [1e-1, 1, 1e1, 1e2, 1e3]
    coef0_range = [0.0,0.1]
    degree_range = [0,1,2,3,4,5,6]

    tuned_parameters = [{'kernel':['rbf','linear'], 'gamma':gamma_range, 'C':C_range, 'class_weight':['balanced']},
        {'kernel':['sigmoid'], 'gamma':gamma_range, 'C':C_range, 'coef0':coef0_range, 'class_weight':['balanced']},
            {'kernel':['poly'], 'gamma':gamma_range, 'C':C_range, 'coef0':coef0_range, 'class_weight':['balanced'], 'degree':degree_range}]

    classifier = SVC(decision_function_shape='ovo')

def lrbinPar():
    global tuned_parameters
    global classifier
    C_range = [1e-3,1e-2,1e-1,1,1e1,1e2,1e3]

    tuned_parameters = [{'solver':['newton-cg', 'lbfgs', 'sag'], 'class_weight':['balanced'], 'C': C_range },
        {'solver':['liblinear'], 'penalty':['l1','l2'], 'class_weight':['balanced'], 'C': C_range},
        {'solver':['sag'], 'penalty':['l2'], 'class_weight':['balanced'], 'C': C_range}]

    classifier = LogisticRegression(multi_class='ovr', n_jobs=-1)

def lrPar():
    global tuned_parameters
    global classifier
    C_range = [1e-3,1e-2,1e-1,1,1e1,1e2,1e3]

    tuned_parameters = [{'solver':['newton-cg', 'lbfgs', 'sag'], 'class_weight':['balanced'], 'C': C_range },
        {'solver':['sag'], 'penalty':['l2'], 'class_weight':['balanced'], 'C': C_range}]

    classifier = LogisticRegression(multi_class='multinomial')

def gbPar():
    global tuned_parameters
    global classifier
    loss_range = ['deviance', 'exponential']
    learningrate_range = [.05, .1, .15, .2] #.1 default
    estimator_range = [100]
    subsmaple_range = [1] #.8
    criterion_range = ['friedman_mse', 'mse', 'mae', ''] #friedman_mse is best
    samplesplit_range = [2] #500
    sampleleaf_range = [1] #50
    weightfraction_range = [0]
    depth_range = [3] #8
    impurity_range = [0]
    maxfeatures_range = ['auto','sqrt','log2'] #sqrt

    tuned_parameters = [{'loss':loss_range,'learning_rate':learningrate_range,'n_estimators':estimator_range,
        'subsample':subsmaple_range,'criterion':criterion_range,'min_samples_split':samplesplit_range,'min_samples_leaf':sampleleaf_range,
        'min_weight_fraction_leaf':weightfraction_range,'max_depth':depth_range,'min_impurity_decrease':impurity_range,'max_features':maxfeatures_range}]

    classifier = GradientBoostingClassifier()
    
    
    tuned_parameters = [{'criterion':criterion_range}]
    classifier = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=500,min_samples_leaf=50,max_depth=8,max_features='sqrt',subsample=0.8,random_state=10)

def gridCV():
    global enable
    global paramDict
    global classifier
    global selectClass
    getData()
    det = input("Enable Detailed Output y/n ? ")
    print("")
    if(det=='y'):
        enable = 1
    else:
        enable=0
        warnings.filterwarnings("ignore")
    print("Classification Mode : ",clfType)
    print("Choose your Classifier for Parameter Tuning")
    clas = int(input("1 - SVC, 2 - SVCMulti, 3 - LR, 4 - LRMulti : "))
    if clas==1 and clfType =='b':
        selectClass = 'SVC'
        svcbinPar()
    elif clas==2 and clfType =='m':
        selectClass = 'SVCMulti'
        svcPar()
    elif clas==3 and clfType =='b':
        selectClass = 'LR'
        lrbinPar()
    elif clas==4 and clfType =='m':
        selectClass = 'LRMulti'
        lrPar()
    else:
        print("Wrong Combination. Please start again")

    scores = ['precision', 'recall']
    print()
    print("#########################################")
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()
        clf = GridSearchCV(classifier, tuned_parameters, cv=5,scoring='%s_macro' % score)
        clf.fit(X_train, y_train)
        paramDict = clf.best_params_
        print("Best parameters set found on development set:")
        print(classifier)
        
        joblib.dump(paramDict, joblib_paramDict)
        joblib.dump(selectClass, joblib_dsClf)
        joblib.dump(selectScaler, joblib_dsScaler)
        
        print("Best Parameters, Classiffier and Scaler Saved")
        print()

        newprint("Grid scores on development set:")
        newprint("")
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            newprint("%0.3f (+/-%0.03f) for %r"
                % (mean, std * 2, params))
        newprint("")

        newprint("Detailed classification report:")
        newprint("")
        newprint("The model is trained on the full development set.")
        newprint("The scores are computed on the full evaluation set.")
        newprint("")
        y_true, y_pred = y_test, clf.predict(X_test)
        newprint(classification_report(y_true, y_pred))
        newprint("")
    print("#########################################")
    print()

def trainData(gesture,dsetnum):
    count2 = 0
    startTime=0
    totalTime = 0
    startTime=datetime.now()
    excount=0
    with click.progressbar(length=dsetnum, label='Collecting Data Points') as itr:
        for z in (itr):
            try:
                
                i=0
                y=0
                j=28
                rcvAry = [0]*407
                count = 0
                        
                with serial.Serial('COM3',115200, timeout=None) as ser:
                    ser.write(b'4')
                    # time.sleep(.1)
                    while i < 28:
                        rcv = ser.readline(20)
                        rcvAry[i] = float(rcv.decode('utf-8'))
                        i +=1
                        count+=1

                for x in range (0, 27):
                    for y in range (x+1, 28):
                        rcvAry[j] = rcvAry[x] - rcvAry[y]
                        j+=1
                        count+=1
                rcvAry[j]=gesture
                count+=1
                np.asarray(rcvAry)

                filename_temp = single_dir+"\\"+filename_base+str(gesture)+'.'+extension
                with open(filename_temp,'a') as csvFile:
                    writer = csv.writer(csvFile,lineterminator='\n')
                    writer.writerow(rcvAry)
                

                count2+=1
                # print("Iteration # ",count2)
                
            except Exception as e:
                print(e)
                print("COM Port Error")
                z=z-1
                excount+=1


    totalTime = datetime.now()-startTime
    print("--------------------------------------------")
    print("Gesture            : ",gesture)
    print("Total Time         : ",totalTime)
    print("Total Run          : ",count2)
    try:
        print("Time/Run           : ",totalTime/count2)
    except Exception as e:
        print("Exceptions Occured : ",e)
    
    print("Total Attributes   :",count)
    print("Exceptions Occured : ",excount)
    print("Iteration Done")
    print("--------------------------------------------")
    print("###############################################################################")
    print("")

def datasetMain():
    global message
    global gesDict
    global gesNum

    gesDict = dict()
    paramDict = dict()

    message=""

    gesNum = int(input("Enter number of Gestures (0 to Skip) : "))
    if(gesNum!=0):
        for gestureNumber in range(0,gesNum):
            gesName = str(input("Enter Gesture Name for "+str(gestureNumber)+" : "))
            gesDict[gestureNumber] = gesName
            message+=str(gestureNumber)+" - "+str(gesName)+", "
        message = message+str(gesNum)+" - GSCV, "+str(gesNum+1)+" - Generate Final Dataset : "
        joblib.dump(gesDict, joblib_gesDict)
        joblib.dump(message, joblib_message)
        joblib.dump(gesNum, joblib_gestNum)
        print()
    else:
        gesDict = joblib.load(joblib_gesDict)
        message = joblib.load(joblib_message)
        gesNum = joblib.load(joblib_gestNum)
        print("Gestures Loaded : ",gesDict)


    while(True):
        global e
        global cdataFile
        global dataFile
        global flistMsg
        global clfType
        flistMsg =""

        inp = int(input(message))
        print()
        if(inp<gesNum):
            print("Selected : "+str(inp)+" - "+gesDict[inp])
            print()
            dsetnum = int(input("Total # of Datasets to Collect : "))

        print()
        print("###############################################################################")
        print()

        temp_list=[]
        nFiles=0
        if inp == gesNum:
            try:
                print("Selected : "+str(inp)+" - GSCV")
                clfType = str(input("Binary or Multi ? b/m : "))
                joblib.dump(clfType, joblib_clfType)

                if clfType =='b':
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
                    print("DataFile Selected : ",dataFile)
                else:
                    dataFile = multi_dir+"\\datasetComplete.csv"
                    print("DataFile Selected : ",dataFile)
                gridCV()
            except Exception as e:
                print("Exception")
                print(e)
        elif inp == gesNum+1:
            print("Selected : "+str(inp)+" - Generate Final Dataset")
            csvconcat()
        else:
            count2= 0
            startTime5 = datetime.now()
            trainData(inp,dsetnum)


datasetMain()