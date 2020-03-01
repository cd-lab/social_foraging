from psychopy import core, visual, gui, data, event
from psychopy.tools.filetools import fromFile, toFile
import time
import numpy as np
import random, math
import os
import sys
import scipy.stats as stats
import pandas as pd

##### expParas #####
def getExpParas():
	expParas = {}
	expParas['conditions'] = ['rich', 'poor']
	expParas['unqHts'] = np.multiply([40, 25, 22, 2.75], 0.7)
	expParas['decsSec'] = 4 * 0.7
	expParas['fbSelfSec'] = 3 * 0.7
	expParas['fbOtherSec'] = 3 * 0.7
	expParas['travelSec'] = 11 * 0.7
	expParas['rwd'] = 2
	expParas['rwdHigh'] = 3
	expParas['rwdLow'] = 1
	expParas['missLoss'] = -2
	expParas['blockSec'] = 3 * 60
	hts_ = {
	'rich' : np.multiply([40, 28, 22, 2.75, 2.75, 2.75, 2.75], 0.7),
	'poor' : np.multiply([40, 28, 28, 28, 28, 22, 2.75], 0.7)
	}
	expParas['hts_'] = hts_
	return expParas

def getSeqs(expParas):
	blockSec = expParas['blockSec']
	iti = expParas['travelSec']
	rwdHigh = expParas['rwdHigh']
	rwdLow = expParas['rwdLow']
	rwd = expParas['rwd']
	unqHts = expParas['unqHts']
	conditions = expParas['conditions']
	hts_ = expParas['hts_']
	# creat new variables
	nCondition = len(conditions)
	chunkSize = len(hts_['rich'])
	nTrialMax = np.ceil(blockSec / iti).astype(int)
	nChunkMax = np.ceil(nTrialMax / chunkSize).astype(int)
	rwds = np.concatenate((np.repeat(rwdLow, chunkSize), np.repeat(rwdHigh, chunkSize)))
	rwdSeq_ = {}
	htSeq_ = {}
	for c in range(nCondition):
		condition = conditions[c]
		hts = hts_[condition]
		# reward sequence 
		junk = []
		for i in range(np.ceil(nChunkMax / 2).astype(int)):
			junk.extend(random.sample(list(rwds), chunkSize * 2)) 
		rwdSeq_[condition] = junk[0 : nTrialMax] # here the data selection doesn't include the tail
		# ht sequence
		junk = []
		for i in range(nChunkMax):
			junk.extend(random.sample(list(hts), chunkSize))
		htSeq_[condition] = junk[0 : nTrialMax]
	outputs = {
		"rwdSeq_" : rwdSeq_,
		"htSeq_" : htSeq_
	}
	return outputs

##### create stimuli #####
def getStims(win, expParas, horCenter):
	verCenter = 0.1
# create stimuli
	trashCan = visual.Rect(win = win, width = 0.3, height = (max(expParas['unqHts']) + 2)  * 0.011,
	units = "height", lineWidth = 4, lineColor = [1, 1, 1], fillColor = [1, 1, 1], pos = (horCenter, verCenter))

	recycleSymbol = visual.ImageStim(win, image="recycle.png", units='height', pos= (horCenter, verCenter),
		size=0.1, ori=0.0, color = "black")
	canPicture = visual.ImageStim(win, image="can.png", units='height', pos= (horCenter, verCenter - 0.1),
		size=0.2, ori=0.0)
	bottlePicture = visual.ImageStim(win, image="bottle.png", units='height', pos= (horCenter, verCenter - 0.1),
		size=0.18, ori=0.0)

	trash = visual.Rect(win=win, width = 0.295, height = 1 * 0.011,
			units = "height", lineWidth = 4, lineColor = [1, 1, 1], fillColor = [0.5, 0.5, 0.5],\
			pos = (horCenter, -(max(expParas['unqHts']) + 2 - 1) / 2 * 0.011 + verCenter))

	# create the traveling time bar 
	whiteTimeBar = visual.Rect(win = win, width = expParas['travelSec'] * 0.03, height = 0.03,
	units = "height", lineWidth = 2, lineColor = [1, 1, 1], fillColor = [1, 1, 1], pos = (horCenter, -0.30 + verCenter))
	timeBarTick = visual.Rect(win = win, pos = (-(expParas['travelSec']/ 2 - expParas['decsSec']) * 0.03 + horCenter, -0.30 + verCenter),
		fillColor = [-1, -1, -1], lineColor = [-1, -1, -1], lineWidth = 2, units = "height", height = 0.05, width = 0.002) # a tick line to indicate when a trashcan will pop up
	timeBarSticker = visual.ImageStim(win, image="recycle.png", units='height', pos= (-(expParas['travelSec']/ 2 - expParas['decsSec']) * 0.03 + horCenter, -0.28 + verCenter + 0.025),
		size=0.03, ori=0.0, color = "black")

	avatar = visual.ImageStim(win, image="avatar.png", units='height', pos= (horCenter, verCenter - 0.1), size = 0.1, ori=0.0)

	# return outputs
	outputs = {'trashCan' : trashCan, 'recycleSymbol' : recycleSymbol, "trash" : trash,\
	'whiteTimeBar' : whiteTimeBar, 'timeBarTick' : timeBarTick, 'timeBarSticker' : timeBarSticker,
	'canPicture' : canPicture, 'bottlePicture' : bottlePicture, "avatar": avatar}
	return(outputs)


def showTrial(win, expParas, expInfo, expHandler, stims, rwdSeq_, htSeq_, ifPrac):
	# constants 
	verCenter = 0.1
	horCenter = 0
	blockSec = expParas['blockSec']

	# define the fucntion to get timeLeftText
	def getTimeLeftLabel(realLeftTime):
		realLeftMin = math.floor(realLeftTime / 60)
		realLeftSec = math.floor(realLeftTime - 60 * realLeftMin)
		timeLeftLabel = "Time Left:" + str(realLeftMin).zfill(2) + ":" + str(realLeftSec).zfill(2)
		return timeLeftLabel

	# define the function to draw the timebar, the remaining blockTime and the total earings 
	def drawTime(frameIdx, realLeftTime, ifPrac):
			whiteTimeBar.draw()
			elapsedSec = frameIdx * expInfo['frameDur']
			leftSec = expParas['travelSec'] - elapsedSec
			blueTimeBar.pos = (- elapsedSec * 0.03 / 2 + horCenter, -0.30 + verCenter)
			blueTimeBar.width = leftSec * 0.03
			blueTimeBar.draw()
			timeBarTick.draw()
			timeBarSticker.draw()
			timeLeftText.text = getTimeLeftLabel(realLeftTime)
			if not ifPrac:
				timeLeftText.draw()

	# parse stims
	trashCan = stims['trashCan']
	trash = stims['trash']
	recycleSymbol = stims['recycleSymbol']
	whiteTimeBar = stims['whiteTimeBar']
	timeBarTick = stims['timeBarTick'] 
	timeBarSticker = stims['timeBarSticker']
	canPicture = stims['canPicture']
	bottlePicture = stims['bottlePicture']

	# calcualte the number of frames for key events
	nFbFrame = math.ceil((expParas['travelSec'] - expParas['decsSec']) / expInfo['frameDur'])
	nDecsFrame = math.ceil(expParas['decsSec'] / expInfo['frameDur'])
	nFbFrameSelf = round(expParas['fbSelfSec'] / expInfo['frameDur'])
			
	# start the task 
	totalEarnings = 0
	if ifPrac:
		nBlock = 1
	else:
		nBlock = 2

	for blockIdx in range(nBlock):
		condition = expParas['conditions'][blockIdx]
		rwdSeq = rwdSeq_[condition]
		htSeq = htSeq_[condition]
		taskTime = blockSec * blockIdx
		trialIdx = 0

		# change the backgroud color 
		if not ifPrac:
			if blockIdx > 0:
				background = visual.ImageStim(win, image="campus2.png", opacity = 0.15,
					interpolate = True, size = [1440, 900], units = 'pix')
			else:
				background = visual.ImageStim(win, image="campus1.png", opacity = 0.15, interpolate = True, size = [1440, 900], units = 'pix')
		else:
			win.color = [0.9764706, 0.5450980, 0.5058824]

		# create the message
		if blockIdx == 0:
			if ifPrac:
				message = visual.TextStim(win=win, ori=0,
				text= 'Press Any Key to Start the Practice', font=u'Arial', bold = True, units='height',\
				pos=[0, 0], height=0.06, color= 'white', colorSpace='rgb') 
			else:
				message = visual.TextStim(win=win, ori=0,
				text= 'Press Any Key to Start', font=u'Arial', bold = True, units='height',\
				pos=[0, 0], height=0.06, color= 'white', colorSpace='rgb') 				
		else:
			if ifPrac:
				message = visual.TextStim(win=win, ori=0,
				text= 'The First Practice Block Ends \n Press Any Key to Start the Second Block Practice in a New Campus', font=u'Arial', bold = True, units='height',\
				pos=[0, 0], height=0.06, color= 'white', colorSpace='rgb')
			else:
				message = visual.TextStim(win=win, ori=0,
				text= 'The First Block Ends \n Press Any Key to Start the Second Block in a New Campus', font=u'Arial', bold = True, units='height',\
				pos=[0, 0], height=0.06, color= 'white', colorSpace='rgb')				


		# clear all events
		event.clearEvents() 
		# wait for any key to start the game
		responded = False
		while responded == False:
			# detect keys
			keysNow = event.getKeys()
			if len(keysNow) > 0:
				responded = True
			message.draw()
			win.flip()

		# record the time left in the block
		realLeftTime = blockSec 
		timeLeftLabel = getTimeLeftLabel(realLeftTime)
		timeLeftText = visual.TextStim(win=win, ori=0,\
		text= timeLeftLabel,\
		font=u'Arial', units='height',\
		pos=[horCenter, -0.35], height= 0.05, color= "white", colorSpace='rgb')

		# create total earnings and trial earnings
		totalEarnText = visual.TextStim(win=win, ori=0,\
		text = "Earned: ", font=u'Arial', units='height',\
		pos=[horCenter, -0.28], height= 0.05, color= "white", colorSpace='rgb')

		trialEarnText = visual.TextStim(win=win, ori=0, font=u'Arial', bold = True, units='height',\
			pos=[horCenter, verCenter + 0.05], height=0.1, color=[-1, -1, -1], colorSpace='rgb') 		

		# create the blue bar
		blueTimeBar = visual.Rect(win = win, height = 0.03,\
		units = "height", lineWidth = 2, lineColor = [1, 1, 1], fillColor = [-0.16078431,  0.36470588,  0.67843137])

		# start plotting
		# plot the first searching time 
		frameIdx = 0
		while frameIdx < nFbFrame:
			if not ifPrac:
				background.draw()
			# draw the timebar
			drawTime(frameIdx, realLeftTime, ifPrac)
			totalEarnText.draw()
			win.flip()
			# update the leftTime
			realLeftTime = realLeftTime - expInfo['frameDur'] 
			frameIdx += 1

		while (not ifPrac and realLeftTime > 0) or (ifPrac and trialIdx < len(expParas['unqHts'])) :
			# if isPrac, terminate the program after experiencing all four possible options
			# reward and handling time for this trial
			scheduledHt = htSeq[trialIdx]
			scheduledRwd = rwdSeq[trialIdx]
            
			# wait for the decision 
			responded = False
			frameIdx = nFbFrame
			# clear all events
			event.clearEvents() 

			while (frameIdx < nDecsFrame + nFbFrame) and realLeftTime > 0:
				if not ifPrac:
					background.draw()
				# draw stimuli
				trashCan.draw()
				trash.height = scheduledHt * 0.011
				trash.pos = (horCenter, -(max(expParas['unqHts']) + 2 - scheduledHt) / 2 * 0.011 + verCenter)
				trash.draw()
				if responded == False:
					recycleSymbol.color = "black"
				else:
					if response == 1:
						recycleSymbol.color = "blue"
					else:
						recycleSymbol.color = "red"
				recycleSymbol.draw()
				# draw the time bar
				drawTime(frameIdx, realLeftTime, ifPrac)
				totalEarnText.draw()
				win.flip()
				# detect keys
				keysNow = event.getKeys(keyList={'k', 'd'}, modifiers=False, timeStamped=True)
				if len(keysNow) > 0:
					responded = True
					response = 1 if keysNow[0][0] == "k" else 0 # 1 for accept, 0 for reject 
					responseRT = (frameIdx - nFbFrame) * expInfo['frameDur'] # take the [0, 1) interval as 0
					responseFrameIdx = frameIdx
					responseBlockTime = (blockSec - realLeftTime) 
				# update the leftTime
				realLeftTime = realLeftTime - expInfo['frameDur'] 
				# update the frame idx and the leftTime
				frameIdx += 1

			# record the response if the trial is missed 
			if responded == False:
				response = -1 # -1 for miss
				responseBlockTime = blockSec - realLeftTime 
				responseRT = np.nan


			# count down if the option is accepted
			if response == 1:
				nCountDownFrame = math.ceil(scheduledHt / expInfo['frameDur'])
				while (frameIdx < nDecsFrame + nFbFrame + nCountDownFrame) and realLeftTime > 0:
					if not ifPrac:
						background.draw()
					trashCan.draw()
					countDownTime = scheduledHt - (frameIdx - nDecsFrame - nFbFrame) * expInfo['frameDur'] # time for the next win flip
					trash.height = countDownTime * 0.011
					trash.pos = (horCenter, -(max(expParas['unqHts']) + 2 - countDownTime) / 2 * 0.011 + verCenter)
					trash.draw()
					recycleSymbol.color = "blue"
					recycleSymbol.draw()
					# draw the left Time
					timeLeftText.text = getTimeLeftLabel(realLeftTime)
					if not ifPrac:
						timeLeftText.draw()
					# draw the total earnings 
					totalEarnText.draw()
					win.flip()
					# update
					frameIdx += 1
					realLeftTime = realLeftTime - expInfo['frameDur'] 

			# trialEarnings and spentHt
			if response == 1:
				trialEarnings = scheduledRwd
				spentHt = scheduledHt
			elif response == 0:
				trialEarnings  = 0
				spentHt = 0
			else:
				trialEarnings = expParas['missLoss']
				spentHt = 0


			# update time and total Earnings
			totalEarnings = totalEarnings + trialEarnings
			blockTime = blockSec - realLeftTime	# block time before searching for the next trashcan
			taskTime = blockSec - realLeftTime + blockIdx * blockSec
			totalEarnText.text = "Earned: " + str(totalEarnings) 

			# update trial earnings 
			trialEarnText.text = str(trialEarnings)

			# save the data before searching for the next trashcan
			expHandler.addData('blockIdx',blockIdx + 1) # since blockIdx starts from 0 
			expHandler.addData('trialIdx',trialIdx + 1)
			expHandler.addData('scheduledHt',scheduledHt)
			expHandler.addData('scheduledRwd',scheduledRwd)
			expHandler.addData('spentHt', spentHt)
			expHandler.addData('responseBlockTime', responseBlockTime)
			expHandler.addData('trialEarnings', trialEarnings)
			expHandler.addData('responseRT', responseRT)
			expHandler.addData('blockTime', blockTime)
			expHandler.nextEntry()

			# give the feedback 

			frameIdx = 0
			while frameIdx < nFbFrame and realLeftTime > 0:	
				if not ifPrac:
					background.draw()
				if frameIdx < nFbFrameSelf:
					if trialEarnings == expParas['rwdHigh']:
						bottlePicture.draw()
					elif trialEarnings == expParas['rwdLow']:
						canPicture.draw()
					trialEarnText.draw()
				drawTime(frameIdx, realLeftTime, ifPrac)
				totalEarnText.draw()
				realLeftTime = realLeftTime - expInfo['frameDur']
				win.flip()
				frameIdx += 1
			# move to the next trial 
			trialIdx = trialIdx + 1
			
	# show the ending massage 
	event.clearEvents() 
	if ifPrac:
		message = visual.TextStim(win=win, ori=0,
		text= 'The Practice Ends \n Press Any Key to Quit', font=u'Arial', bold = True, units='height',\
		pos=[0, 0], height=0.06, color= 'white', colorSpace='rgb') 
	else:
		message = visual.TextStim(win=win, ori=0,
		text= 'The Experiment Ends \n Press Any Key to Quit', font=u'Arial', bold = True, units='height',\
		pos=[0, 0], height=0.06, color= 'white', colorSpace='rgb') 		
	# wait for any key to quit 
	responded = False
	while responded == False:
		# detect keys
		keysNow = event.getKeys()
		if len(keysNow) > 0:
			responded = True
		message.draw()
		win.flip()

	# save the total earnings 
	trialOutput = {'expHandler':expHandler, 'totalEarnings': totalEarnings} 
	win.close()
	return trialOutput


def showTrialSocial(win, expParas, expInfo, expHandler, stims, rwdSeq_, htSeq_, ifPrac):
	# constants 
	verCenter = 0.1
	horCenter = 0
	blockSec = expParas['blockSec']

	# feedback 
	# refData = pd.read_csv("reference_data/reference.csv", header = 0)
	# refData['taskTime'] = refData['blockTime'] + (refData['blockIdx'] - 1) * expParas['blockSec']
	refData = pd.read_csv("reference_data/reference_group.csv", header = 0)
	refData['taskTime'] = refData['responseBlockTime'] + (refData['condition'] == "poor") * expParas['blockSec']
	nSub = len(np.unique(refData['id']))

	# define the fucntion to get timeLeftText
	def getTimeLeftLabel(realLeftTime):
		realLeftMin = math.floor(realLeftTime / 60)
		realLeftSec = math.floor(realLeftTime - 60 * realLeftMin)
		timeLeftLabel = "Time Left:" + str(realLeftMin).zfill(2) + ":" + str(realLeftSec).zfill(2)
		return timeLeftLabel

	# define the function to draw the timebar, the remaining blockTime and the total earings 
	def drawTime(frameIdx, realLeftTime, ifPrac):
			whiteTimeBar.draw()
			elapsedSec = frameIdx * expInfo['frameDur']
			leftSec = expParas['travelSec'] - elapsedSec
			blueTimeBar.pos = (- elapsedSec * 0.03 / 2 + horCenter, -0.30 + verCenter)
			blueTimeBar.width = leftSec * 0.03
			blueTimeBar.draw()
			timeBarTick.draw()
			timeBarSticker.draw()
			timeLeftText.text = getTimeLeftLabel(realLeftTime)
			if not ifPrac:
				timeLeftText.draw()

	# parse stims
	trashCan = stims['trashCan']
	trash = stims['trash']
	recycleSymbol = stims['recycleSymbol']
	whiteTimeBar = stims['whiteTimeBar']
	timeBarTick = stims['timeBarTick'] 
	timeBarSticker = stims['timeBarSticker']
	canPicture = stims['canPicture']
	canPicture.pos = [horCenter - 0.1, verCenter - 0.1]
	bottlePicture = stims['bottlePicture']
	bottlePicture.pos = [horCenter - 0.1, verCenter - 0.1]
	avatar = stims['avatar']
	avatar.pos = [horCenter + 0.1, verCenter - 0.1]
	# calcualte the number of frames for key events
	nFbFrame = math.ceil((expParas['travelSec'] - expParas['decsSec']) / expInfo['frameDur'])
	nDecsFrame = math.ceil(expParas['decsSec'] / expInfo['frameDur'])
	nFbFrameSelf = round(expParas['fbSelfSec'] / expInfo['frameDur'])
	nFbFrameOther = round(expParas['fbOtherSec'] / expInfo['frameDur'])

	# start the task 
	totalEarnings = 0
	if ifPrac:
		nBlock = 1
	else:
		nBlock = 2

	for blockIdx in range(nBlock):
		condition = expParas['conditions'][blockIdx]
		rwdSeq = rwdSeq_[condition]
		htSeq = htSeq_[condition]
		taskTime = blockSec * blockIdx
		trialIdx = 0

		# change the backgroud color 
		if not ifPrac:
			if blockIdx > 0:
				background = visual.ImageStim(win, image="campus2.png", opacity = 0.15,
					interpolate = True, size = [1440, 900], units = 'pix')
			else:
				background = visual.ImageStim(win, image="campus1.png", opacity = 0.15, interpolate = True, size = [1440, 900], units = 'pix')
		else:
			win.color = [0.9764706, 0.5450980, 0.5058824]

		# create the message
		if blockIdx == 0:
			if ifPrac:
				message = visual.TextStim(win=win, ori=0,
				text= 'Press Any Key to Start the Practice', font=u'Arial', bold = True, units='height',\
				pos=[0, 0], height=0.06, color= 'white', colorSpace='rgb') 
			else:
				message = visual.TextStim(win=win, ori=0,
				text= 'Press Any Key to Start', font=u'Arial', bold = True, units='height',\
				pos=[0, 0], height=0.06, color= 'white', colorSpace='rgb') 				
		else:
			if ifPrac:
				message = visual.TextStim(win=win, ori=0,
				text= 'The First Practice Block Ends \n Press Any Key to Start the Second Block Practice in a New Campus', font=u'Arial', bold = True, units='height',\
				pos=[0, 0], height=0.06, color= 'white', colorSpace='rgb')
			else:
				message = visual.TextStim(win=win, ori=0,
				text= 'The First Block Ends \n Press Any Key to Start the Second Block in a New Campus', font=u'Arial', bold = True, units='height',\
				pos=[0, 0], height=0.06, color= 'white', colorSpace='rgb')				


		# clear all events
		event.clearEvents() 
		# wait for any key to start the game
		responded = False
		while responded == False:
			# detect keys
			keysNow = event.getKeys()
			if len(keysNow) > 0:
				responded = True
			message.draw()
			win.flip()

		# record the time left in the block
		realLeftTime = blockSec 
		timeLeftLabel = getTimeLeftLabel(realLeftTime)
		timeLeftText = visual.TextStim(win=win, ori=0,\
		text= timeLeftLabel,\
		font=u'Arial', units='height',\
		pos=[horCenter, -0.35], height= 0.05, color= "white", colorSpace='rgb')

		# create the total earnings and trial earnings
		totalEarnText = visual.TextStim(win=win, ori=0,\
			text = "Earned: ", font=u'Arial', units='height',\
			pos=[horCenter, -0.28], height= 0.05, color= "white", colorSpace='rgb')


		# create the blue bar
		blueTimeBar = visual.Rect(win = win, height = 0.03,\
		units = "height", lineWidth = 2, lineColor = [1, 1, 1], fillColor = [-0.16078431,  0.36470588,  0.67843137])
        
        # create feedback 
		trialEarnText = visual.TextStim(win=win, ori=0, font=u'Arial', bold = True, units='height',\
		pos=[horCenter - 0.1, verCenter + 0.05], height=0.1, color=[-1, -1, -1], colorSpace='rgb') 	
		trialEarnOtherText = visual.TextStim(win=win, ori=0, font=u'Arial', bold = True, units='height',\
		pos=[horCenter + 0.1, verCenter + 0.05], height=0.1, color=[1, -1, -1], colorSpace='rgb') 		

		# plot the first searching time 
		frameIdx = 0
		while frameIdx < nFbFrame:
			if not ifPrac:
				background.draw()
			# draw the timebar
			drawTime(frameIdx, realLeftTime, ifPrac)
			totalEarnText.draw()
			win.flip()
			# update the leftTime
			realLeftTime = realLeftTime - expInfo['frameDur'] 
			frameIdx += 1

		while (not ifPrac and realLeftTime > 0) or (ifPrac and trialIdx < len(expParas['unqHts'])) :
			# if isPrac, terminate the program after experiencing all four possible options
			# reward and handling time for this trial
			scheduledHt = htSeq[trialIdx]
			scheduledRwd = rwdSeq[trialIdx]
            
			# wait for the decision 
			responded = False
			frameIdx = nFbFrame
			# clear all events
			event.clearEvents() 

			while (frameIdx < nDecsFrame + nFbFrame) and realLeftTime > 0:
				if not ifPrac:
					background.draw()
				# draw stimuli
				trashCan.draw()
				trash.height = scheduledHt * 0.011
				trash.pos = [horCenter, -(max(expParas['unqHts']) + 2 - scheduledHt) / 2 * 0.011 + verCenter]
				trash.draw()
				if responded == False:
					recycleSymbol.color = "black"
				else:
					if response == 1:
						recycleSymbol.color = "blue"
					else:
						recycleSymbol.color = "red"
				recycleSymbol.draw()
				# draw the time bar
				drawTime(frameIdx, realLeftTime, ifPrac)
				totalEarnText.draw()
				win.flip()
				# detect keys
				keysNow = event.getKeys(keyList={'k', 'd'}, modifiers=False, timeStamped=True)
				if len(keysNow) > 0:
					responded = True
					response = 1 if keysNow[0][0] == "k" else 0 # 1 for accept, 0 for reject 
					responseRT = (frameIdx - nFbFrame) * expInfo['frameDur'] # take the [0, 1) interval as 0
					responseFrameIdx = frameIdx
					responseBlockTime = (blockSec - realLeftTime) 
				# update the leftTime
				realLeftTime = realLeftTime - expInfo['frameDur'] 
				# update the frame idx and the leftTime
				frameIdx += 1

			# record the response if the trial is missed 
			if responded == False:
				response = -1 # -1 for miss
				responseBlockTime = blockSec - realLeftTime 
				responseRT = np.nan


			# count down if the option is accepted
			if response == 1:
				nCountDownFrame = math.ceil(scheduledHt / expInfo['frameDur'])
				while (frameIdx < nDecsFrame + nFbFrame + nCountDownFrame) and realLeftTime > 0:
					if not ifPrac:
						background.draw()
					trashCan.draw()
					countDownTime = scheduledHt - (frameIdx - nDecsFrame - nFbFrame) * expInfo['frameDur'] # time for the next win flip
					trash.height = countDownTime * 0.011
					trash.pos = [horCenter, -(max(expParas['unqHts']) + 2 - countDownTime) / 2 * 0.011 + verCenter]
					trash.draw()
					recycleSymbol.color = "blue"
					recycleSymbol.draw()
					# draw the left Time
					timeLeftText.text = getTimeLeftLabel(realLeftTime)
					if not ifPrac:
						timeLeftText.draw()
					# draw the total earnings 
					totalEarnText.draw()
					win.flip()
					# update
					frameIdx += 1
					realLeftTime = realLeftTime - expInfo['frameDur'] 

			# trialEarnings and spentHt
			if response == 1:
				trialEarnings = scheduledRwd
				spentHt = scheduledHt
			elif response == 0:
				trialEarnings  = 0
				spentHt = 0
			else:
				trialEarnings = expParas['missLoss']
				spentHt = 0


			# update time and total Earnings
			totalEarnings = totalEarnings + trialEarnings
			blockTime = blockSec - realLeftTime	# block time before searching for the next trashcan
			preTaskTime = taskTime # save the timepoint on the end of the previous block 
			taskTime = blockSec - realLeftTime + blockIdx * blockSec # update taskTime
			totalEarnText.text = "Earned: " + str(totalEarnings) 

			# calculate how many points the other player has earned during the current trial
			trialEarningsOther = round(np.sum(refData[(refData['taskTime'] < taskTime) & (refData['taskTime'] >= preTaskTime)]['trialEarnings']) / nSub, 1)
			

			# update self earnings and other earnings 
			trialEarnText.text = str(trialEarnings)
			trialEarnOtherText.text = str(trialEarningsOther)

			# save the data before searching for the next trashcan
			expHandler.addData('blockIdx',blockIdx + 1) # since blockIdx starts from 0 
			expHandler.addData('trialIdx',trialIdx + 1)
			expHandler.addData('scheduledHt',scheduledHt)
			expHandler.addData('scheduledRwd',scheduledRwd)
			expHandler.addData('spentHt', spentHt)
			expHandler.addData('responseBlockTime', responseBlockTime)
			expHandler.addData('trialEarnings', trialEarnings)
			expHandler.addData('responseRT', responseRT)
			expHandler.addData('blockTime', blockTime)
			expHandler.nextEntry()

			# give the feedback 
			frameIdx = 0
			while frameIdx < nFbFrame and realLeftTime > 0:	
				if not ifPrac:
					background.draw()
				if frameIdx < nFbFrameOther + nFbFrameSelf:
					if trialEarnings == expParas['rwdHigh']:
						bottlePicture.draw()
					elif trialEarnings == expParas['rwdLow']:
						canPicture.draw()
					trialEarnText.draw()
				if (frameIdx < nFbFrameOther + nFbFrameSelf) & (nFbFrameSelf <= frameIdx):
					trialEarnOtherText.draw()
					avatar.draw() 
				drawTime(frameIdx, realLeftTime, ifPrac)
				totalEarnText.draw()
				realLeftTime = realLeftTime - expInfo['frameDur']
				win.flip()
				frameIdx += 1
			# move to the next trial 
			trialIdx = trialIdx + 1
			
	# show the ending massage 
	event.clearEvents() 
	if ifPrac:
		message = visual.TextStim(win=win, ori=0,
		text= 'The Practice Ends \n Press Any Key to Quit', font=u'Arial', bold = True, units='height',\
		pos=[0, 0], height=0.06, color= 'white', colorSpace='rgb') 
	else:
		message = visual.TextStim(win=win, ori=0,
		text= 'The Experiment Ends \n Press Any Key to Quit', font=u'Arial', bold = True, units='height',\
		pos=[0, 0], height=0.06, color= 'white', colorSpace='rgb') 		
	# wait for any key to quit 
	responded = False
	while responded == False:
		# detect keys
		keysNow = event.getKeys()
		if len(keysNow) > 0:
			responded = True
		message.draw()
		win.flip()

	# save the total earnings 
	trialOutput = {'expHandler':expHandler, 'totalEarnings': totalEarnings} 
	win.close()
	return trialOutput

