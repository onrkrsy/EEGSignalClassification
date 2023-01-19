import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import sleep
import os
import csv
##import pyeeg
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
print("lets Start")

def fList(dirx):
    tempx = []
    for file in os.listdir(dirx):
        fl = dirx + file
        tempx.append(fl)
    return sorted(tempx)

# import each and every file
filesA = fList("./src/Datas/Z/")
filesB = fList("./src/Datas/O/")
filesC = fList("./src/Datas/N/")
filesD = fList("./src/Datas/F/")
filesE = fList("./src/Datas/S/")
print(filesE)


# create small tables
st = 'A'
def sTable(tempY):
    tx = []
    for i in range(len(tempY)):
        x = pd.read_table(tempY[i], header=None)
        x.columns = [st + str(i)]
        tx.append(x)
    return tx

ta = sTable(filesA)
tb = sTable(filesB)
tc = sTable(filesC)
td = sTable(filesD)
te = sTable(filesE)



print(ta[0])


# create Big tables
def table(table):
    big_table = None
    for ta in table:
        big_table = pd.concat([big_table, ta],axis=1)
    return big_table

bigA = table(ta)
bigB = table(tb)
bigC = table(tc)
bigD = table(td)
bigE = table(te)
head = list(bigA.columns.values)

#print(len(bigB.columns))
print(bigB)

def creat_mat(mat):
    matx = np.zeros((len(mat),(len(head))))
    for i in range(len(head)):
        matx[:,i] = mat[head[i]]
        #sleep(0.01)
    return matx

matA =creat_mat(bigA)
matB = creat_mat(bigB) # : refers to healthy data
matC = creat_mat(bigC) # : refers to Inter-ictal (transition between healthy to seizure)
matD =creat_mat(bigD)
matE = creat_mat(bigE) # : of ictal or seizures
#print(matB)
matA = np.nan_to_num(matA)
matB = np.nan_to_num(matB) # matB[:,0] --- > channel 0, matB[:,1] --- > channel 1 like that
matC = np.nan_to_num(matC)
matD = np.nan_to_num(matD)
matE = np.nan_to_num(matE)
#print(matB)

# 4097 data point per channel
# 173.61 Hz sample rate and there are 4097 data point for each channel
# total 100 channel are their
# 4097/173.61 = 23.59 sec
# the raw data from one of the channels for the 23.59 sec

hlopen, = plt.plot(matA[0],label='healthy-open')
hl,      = plt.plot(matB[0],label='healthy')
trans,   = plt.plot(matC[0],label='Inter-ictal')
notseizure, = plt.plot(matD[0],label='not in seizure')
seizure, = plt.plot(matE[0],label='seizures')

plt.legend(handles=[hlopen,hl,trans,notseizure,seizure])
plt.savefig("fig1.png")


result = np.concatenate((matA, matB, matC, matD, matE))
with open('test.csv', 'w') as fp:
   writer = csv.writer(fp, delimiter=';')
   for row in result:
      writer.writerow(row)