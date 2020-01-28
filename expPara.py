import pickle
import numpy as np
import math

# reward paras
rwd = 2
highRwd = 3.5
lowRwd = 0.5

# time paras
iti = 4
conditions = ["rich", "poor"]
nCondition = len(conditions)
hts_ = {"rich": np.array([18, 13, 10, 2, 2, 2, 2]),\
"poor" : np.array([18, 18, 18, 18, 13, 10, 2])}
unqHts = np.unique(hts_['poor'])
nUnqHt = len(unqHts)
chunkSize = len(hts_['poor'])

# calculate the optimal longRunRate 
# Accept hts <= threshod

# the customized function
def getLongRunRate(ht):
	totalRwd = sum(hts <= ht) * rwd
	totalTime = iti * chunkSize + sum(hts[hts <= ht])
	longRunRate = totalRwd / totalTime
	return(longRunRate)


getLongRunRates = np.vectorize(getLongRunRate)

optimLongRunRate_ = {}
optimMaxAcpHt_ = {}
for i in range(nCondition):
	condition = conditions[i]
	hts = hts_[condition]
	longRunRates = getLongRunRates(hts)
	optimLongRunRate_[condition] = max(longRunRates)
	optimMaxAcpHt_[condition] = max(unqHts[(rwd / unqHts) >= optimLongRunRate_[condition]])

# block paras
blockSec = 600
nTrialMax = blockSec / iti
nChunkMax = math.ceil(nTrialMax / chunkSize)

# tGrid paras
tGridGap = 1


# save all the objects 
expParas = {
'rwd' : rwd,
'iti':iti,
'conditions' : conditions,
'nCondition' : nCondition,
'hts_' : hts_,
'unqHts' : unqHts,
'nUnqHt' : nUnqHt,
'chunkSize' : chunkSize,
'optimLongRunRate_' : optimLongRunRate_,
'optimMaxAcpHt_' : optimMaxAcpHt_,
'blockSec' : blockSec,
'nTrialMax' : nTrialMax,
"nChunkMax" : nChunkMax, 
"tGridGap" : tGridGap
}

with open('expParas.dic', 'wb') as expParasFile:
	pickle.dump(, expParasFile)






import pickle
with open('expParas.dic', 'rb') as expParasFile:
    expParas = pickle.load(expParasFile)

print(expParas)

