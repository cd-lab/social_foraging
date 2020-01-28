from psychopy import core, visual, gui, data, event
from psychopy.tools.filetools import fromFile, toFile
import time
import numpy as np
import random, math
import os
import sys
import scipy.stats as stats
import numpy.random as rand
# customized package 
import subFxs as sf

# collect participant info
expName = 'Social_Forage'
expInfo = {'participant':'test', 'social_info_condition':'0'}
dlg = gui.DlgFromDict(dictionary=expInfo, title=expName)
if dlg.OK == False:
    core.quit() 
expInfo['expName'] = expName
expInfo['date'] = time.strftime("%d%m%Y")

# set the working path
wkPath = os.getcwd()
os.chdir(wkPath)

# create the data folder
dataPath = wkPath + os.sep + "data"
if not os.path.exists(dataPath):
    os.mkdir(dataPath)

# file name stem, without the extension
fileName =  dataPath + os.sep + u'%s_%s_%s' %(expInfo['participant'], expInfo['date'], expInfo['social_info_condition'])
 
# use an ExperimentHandler to handle saving data
thisExp = data.ExperimentHandler(name=expName, version='',
    extraInfo=expInfo, runtimeInfo=None, originPath=None,
    savePickle=True, saveWideText=True, dataFileName=fileName)

# experiment paras 
expParas = sf.getExpParas()

# setup the Window
win = visual.Window(
    size=(600, 400), fullscr=False, screen=0,
    allowGUI=False, allowStencil=False,
    monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
    blendMode='avg', useFBO=True, pos = [0, 0])

# store frame rate of monitor if we can measure it successfully
expInfo['frameRate']=win.getActualFrameRate()
print('measured frame rate: ')
print(expInfo['frameRate'])
if expInfo['frameRate']!=None:
    expInfo['frameDur'] = 1/round(expInfo['frameRate']) 
else:
    expInfo['frameDur'] = 1/ (60.0) # couldn't get a reliable measure so guess
expInfo['frameDur'] = expInfo['frameDur']



# create stimuli
stims = sf.getStimsSocial(expParas, win)

# # generate the reward sequences and the handling time sequences 
seqResults = sf.getSeqs(expParas)
rwdSeq_ = seqResults['rwdSeq_']
htSeq_ = seqResults['htSeq_']

isSocial = False
    
# for a specific trial
for c in range(len(expParas['conditions'])):
    condition = expParas['conditions'][c]
    rwdSeq = rwdSeq_[condition]
    htSeq = htSeq_[condition]
    taskTime = expParas['blockSec'] * c 
    sf.showTrialSocial(win, expParas, expInfo, thisExp, stims, htSeq, rwdSeq, taskTime)


 





















