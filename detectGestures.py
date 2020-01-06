from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np
from sklearn.externals import joblib
import csv
import serial
from sklearn.externals import joblib
import os
import warnings

os.system('cls')

#Directory Lists
#---------------------------------------------------
diradr = os.getcwd()
curPath = diradr.replace(os.sep, '\\')

main_dir = curPath+'\\Data'
single_dir = curPath+'\\Data\\dataSetSingle'
binary_dir = curPath+'\\Data\\dataSetBinary'
multi_dir = curPath+'\\Data\\dataSetMulti'
realtime_dir = curPath+'\\Data\\RealtimeData'
#---------------------------------------------------

joblib_gesDict = "gestDict.elxr"
joblib_gesTrain = "gesTrain.elxr"
joblib_detType = "detType.elxr"
joblib_gestNum = "gesturenumber.elxr"

gesDict = joblib.load(joblib_gesDict)
gesTrain = joblib.load(joblib_gesTrain)
detType = joblib.load(joblib_detType)
gesNum = joblib.load(joblib_gestNum)


message = ''
global g1
global g2
global g1name
global g2name

if(detType=='b'):
    g1 = int(gesTrain[0])
    g2 = int(gesTrain[1])
    g1name = gesDict[g1]
    g2name = gesDict[g2]

    print("Gestures being Detected : "+str(g1)+
       " - "+g1name+" AND "+str(g2)+" - "+g2name)

else:
    for gestureNumber in range(0,gesNum):
        gesName = gesDict[gestureNumber]
        message+=str(gestureNumber)+" - "+str(gesName)+", "
    print("Gestures being Detected : "+message)

#Read saved Model and Scaler
joblib_modelfile = "joblib_model.pkl"
joblib_scalerfile = "joblib_scaler.pkl"
model = joblib.load(joblib_modelfile)
scaler = joblib.load(joblib_scalerfile)
print("Load Model and Scaler Complete")
print("Model  : ", model)
print("Scaler : ", scaler)

#Custom Print Job
def newprint(pridat,data=None):
    if(enable==1):
        print(pridat,data)

errcount = 0

def predictData():
    global comErr
    global errcount
    i=0
    y=0
    j=28
    rcvAry = [0]*406
    count = 0


    try:
        with serial.Serial('COM4', 115200, timeout=1) as ser:
            ser.write(b'D')
            while i < 28:
                rcv = ser.readline(20)
                rcvAry[i] = float(rcv.decode('utf-8'))
                newprint(rcvAry[i])
                i+=1
                count+=1
                newprint("Total index (COM Data) : ", count)
        comErr = 0
        errcount = 0
    except:
        if(errcount!=1):
            print("COM Port Error")
            errcount = 1
        comErr = 1

    if(comErr!=1):
        for x in range (0, 27):
            for y in range (x+1, 28):
                rcvAry[j] = rcvAry[x] - rcvAry[y]
                newprint(rcvAry[j])
                j+=1
                count+=1
                newprint("Total index (COM + Computed Data) : ", count)

        rcvAry = np.asarray(rcvAry)

        filename = realtime_dir+"datasetComplete.csv"
        # Writing realtime data 2to a file
        with open(filename, 'a') as csvFile:
            writer = csv.writer(csvFile,lineterminator='\n')
            writer.writerow(rcvAry)

        #Transforming Real Time Data
        rcvAry = rcvAry.reshape(-1,406)
        rcvAry = scaler.transform(rcvAry)

        #Predict and Score
        predict = model.predict(rcvAry)
        print ("Gesture Detected : "+str(predict[0])+" - "+str(gesDict[int(predict[0])]))

while(True):
    global enable
    det = input("Enable Detailed Output ? y/n ")
    if det=='y':
        enable = 1
    else:
        enable = 0
        warnings.filterwarnings("ignore")
    method = int(input("1 - Single Prediction, 2 - Continuous Prediction : "))
    print()

    if (method==1):
        predictData()
    else:
        while(True):
            predictData()