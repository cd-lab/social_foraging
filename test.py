from psychopy import core, visual, gui, data, event
from psychopy.tools.filetools import fromFile, toFile
import time
import numpy as np
import random, math
import os
import sys
import pandas as pd
import scipy.stats as stats
import numpy.random as rand
# customized package 
import subFxs as sf

# set the random seed
seed = random.randint(1,10000)
random.seed(seed)

# set the working directory
wkPath = os.getcwd()
os.chdir(wkPath)

# create the data folder
dataPath = wkPath + os.sep + "data"
if not os.path.exists(dataPath):
    os.mkdir(dataPath)

# get the experiment paras 
expParas = sf.getExpParas()

# collect participant info
expName = 'Social_Forage'
expInfo = {'participant':'test', 'social_info_condition':'0'}
dlg = gui.DlgFromDict(dictionary=expInfo, title=expName)
if dlg.OK == False:
    core.quit() 
expInfo['expName'] = expName
expInfo['date'] = time.strftime("%d%m%Y")

# setup the Window
win = visual.Window(fullscr=True, screen=0,
    allowGUI=False, allowStencil=False, size = [1440, 900], units = 'pix',
    monitor='testMonitor', colorSpace='rgb',
    blendMode='avg', useFBO=True, pos = [0, 0])


# create stimuli
stims = sf.getStims(win, expParas, 0)
# save the frame rate of the monitor if we can measure it
expInfo['frameRate']=win.getActualFrameRate()
print('measured frame rate: ')
print(expInfo['frameRate'])
if expInfo['frameRate']!=None:
    expInfo['frameDur'] = 1/round(expInfo['frameRate'])
else:
    expInfo['frameDur'] = 1/60.0 # couldn't get a reliable measure so guess
expInfo['frameDur'] = expInfo['frameDur']

# create the experiment handlers to save data
fileName = dataPath + os.sep + u'%s' %(expInfo['participant'])
headerName = dataPath + os.sep + u'%s_header' %(expInfo['participant'])
thisHeader = data.ExperimentHandler(name = expName, version = "",\
runtimeInfo = None, originPath = None, savePickle = False,\
saveWideText = True, dataFileName = headerName)
thisExp = data.ExperimentHandler(name = expName, version = '',
    runtimeInfo=None, originPath=None,
    savePickle=False, saveWideText=True, dataFileName=fileName)

# set the global event
# clear the global event keys 
event.globalKeys.clear()
def quitFun():
    # save the experiment data
    thisExp.saveAsWideText(fileName+'.csv')
    thisExp.abort() 
    # add entries to the header file 
    thisHeader.addData("subId", expInfo['participant'])
    thisHeader.addData("socialCondition", expInfo['social_info_condition'])
    thisHeader.addData("date", expInfo['date'])
    thisHeader.addData("frameDur", expInfo['frameDur'])
    thisHeader.addData("frameRate", expInfo['frameRate'])
    thisHeader.addData("seed", seed)
    try:
        expData = pd.read_csv(fileName + ".csv")
        totalEarnings = sum(expData['trialEarnings'])
    except pd.errors.EmptyDataError:
        totalEarnings = 0
    totalPayments = totalEarnings / 20 
    thisHeader.addData("totalPayments", totalPayments)
    thisHeader.saveAsWideText(headerName+'.csv')
    thisHeader.abort()
    # close everything
    win.close()
    core.quit()
event.globalKeys.add(key = "q", func = quitFun)

# run the experiment 
seqResults = sf.getSeqs(expParas)
rwdSeq_ = seqResults['rwdSeq_']
htSeq_ = seqResults['htSeq_']
trialOutput = sf.showTrial(win, expParas, expInfo, thisExp, stims, rwdSeq_, htSeq_, False)
thisExp = trialOutput['expHandler']

# add data to the headerFile 
totalPayments = trialOutput['totalEarnings'] / 20 
thisHeader.addData("subId", expInfo['participant'])
thisHeader.addData("socialCondition", expInfo['social_info_condition'])
thisHeader.addData("date", expInfo['date'])
thisHeader.addData("frameDur", expInfo['frameDur'])
thisHeader.addData("frameRate", expInfo['frameRate'])
thisHeader.addData("totalPayments", totalPayments)
thisHeader.addData("seed", seed)

# quit the experiment 
thisHeader.saveAsWideText(headerName + '.csv')
thisHeader.abort()
thisExp.saveAsWideText(fileName +'.csv')
thisExp.abort() # should save the data at the same time
win.close()
core.quit()



 





















