import serial
import os


14
def getData():
    i=0
    rcvAry = [0]*407
    count = 0

    with serial.Serial('COM4', 115200, timeout=1) as ser:
        ser.write(b'D')
        while i < 28:
            rcv = ser.readline(20)
            rcvAry[i] = float(rcv.decode('utf-8'))
            i +=1
            count+=1
    print (rcvAry[pnum-1])

while(True):
    global pnum
    global counter
    pnum = int(input("Pair Number : "))
    getData()