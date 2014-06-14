import re
import os
import numpy as np

def savelabel(guess,savepath, filename): # without test
    os.system('mkdir -p %s'% savepath)
    savename = os.path.join(savepath,os.path.split(filename)[1])
    print savename
    with open(savename,'w') as fid:
        for iS in guess:
            stringVec = [str(x) for x in iS[0]]
            string = ', '.join(stringVec)
            string = string +'\n'
            print string
            fid.writelines(string)

def readmfc(filename):
    with open(filename,'r') as fl:
        line = fl.readline()
        line=line.replace('\n','')
        data =np.array(re.split(' ',line))
        data = data.astype(np.float)
        while line:
            d =np.array(re.split(' ',line))
            data = np.vstack((data,d.astype(np.float32)));
            line = fl.readline()
            line=line.replace('\n','')
    return data

def loadmfc(filename):
    with open(filename,'r') as fl:
        line = fl.readline()
        line=line.replace('\n','')
        line=line.replace('wav','mft')
        data = np.array([readmfc(line)])
        fname = [line]
        while line:
            data = np.vstack((data,[readmfc(line)]))
            line = fl.readline()
            line=line.replace('\n','')
            line=line.replace('wav','mft')
            fname.append(line)
    return (data, fname)


def readlab(filename):
    sr = 16000
    frameInc = 0.01
    mfc = readmfc(filename.replace('PHN','mft'))
    size = list(mfc.shape)

    lab = np.array(np.hstack((np.zeros(tuple([size[0],1])),np.ones(tuple([size[0],1])))))
    with open(filename,'r') as fl:
        line = fl.readline()
        line=line.replace('\n','')
        data =np.array(re.split(' ',line))
        if data[2] == 'cough':
            xmin = int(float(data[0])/sr/frameInc)
            xmax = int(float(data[1])/sr/frameInc)
            lab[xmin:xmax,0]=1
            lab[xmin:xmax,1]=0

        while line:
            data =np.array(re.split(' ',line))
            line = fl.readline()
            line=line.replace('\n','')
            if data[2] == 'cough':
                xmin = int(float(data[0])/sr/frameInc)
                xmax = int(float(data[1])/sr/frameInc)
                lab[xmin:xmax,0]=1
                lab[xmin:xmax,1]=0
    return lab

def loadlabel(filename):
    with open(filename,'r') as fl:
        line = fl.readline()
        line=line.replace('\n','')
        line=line.replace('wav','PHN')
        label= np.array([readlab(line)])
        fname = [line]
        while line:
            label= np.vstack((label,[readlab(line)]))
            fname.append(line)
            line = fl.readline()
            line=line.replace('\n','')
            line=line.replace('wav','PHN')

    return (label,fname)

