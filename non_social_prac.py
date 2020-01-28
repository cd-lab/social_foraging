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
expInfo = {'participant':'test', 'social_info_condition':0}
dlg = gui.DlgFromDict(dictionary=expInfo, title=expName)
if dlg.OK == False:
    core.quit() 
expInfo['expName'] = expName
expInfo['date'] = time.strftime("%d%m%Y")

# setup the Window
win = visual.Window(fullscr=True, screen=0,
    allowGUI=False, allowStencil=False, size = [1280, 800], units = 'pix',
    monitor='testMonitor', color = [0.9764706, 0.5450980, 0.5058824], colorSpace='rgb',
    blendMode='avg', useFBO=True, pos = [0, 0])

# create stimuli
stims = sf.getStims(win, expParas)

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
fileName = dataPath + os.sep + u'prac_%s' %(expInfo['participant'])
headerName = dataPath + os.sep + u'prac_%s_header' %(expInfo['participant'])
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
    totalPayments = 0
    thisHeader.addData("totalPayments", totalPayments)
    thisHeader.saveAsWideText(headerName+'.csv')
    thisHeader.abort()
    # close everything
    win.close()
    core.quit()
event.globalKeys.add(key = "q", func = quitFun)

# run the experiment 
expParas = sf.getExpParas()
rwdSeq_ = {}
htSeq_ = {}
rwdSeq_['poor'] = [expParas['rwdHigh'], expParas['rwdLow'], expParas['rwdHigh'], expParas['rwdLow']]
random.shuffle(rwdSeq_['poor'])
rwdSeq_['rich'] = rwdSeq_['poor']
htSeq_['poor'] = expParas['unqHts']
random.shuffle(htSeq_['poor'])
htSeq_['rich'] = htSeq_['poor']
trialOutput = sf.showTrial(win, expParas, expInfo, thisExp, stims, rwdSeq_, htSeq_, True)
thisExp = trialOutput['expHandler']

# add data to the headerFile 
totalPayments = 0
thisHeader.addData("subId", expInfo['participant'])
thisHeader.addData("socialCondition", expInfo['social_info_condition'])
thisHeader.addData("date", expInfo['date'])
thisHeader.addData("frameDur", expInfo['frameDur'])
thisHeader.addData("frameRate", expInfo['frameRate'])
thisHeader.addData("totalPayments", totalPayments)
thisHeader.addData("seed", seed)

# save data
thisExp.saveAsWideText(fileName+'.csv')
thisHeader.saveAsWideText(headerName+'.csv')

# quit the experiment 
thisHeader.saveAsWideText(headerName + '.csv')
thisHeader.abort()
thisExp.saveAsWideText(fileName +'.csv')
thisExp.abort() # should save the data at the same time
win.close()
core.quit()


 





















