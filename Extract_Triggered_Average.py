'''
Script to extract trigger - synchronized average raw data from binary neuropixel AP data
@author Moritz Drüke, 2024, https://github.com/MDrueke/NPX_Extract_Triggered_Average/tree/main

Parameters (User input, see below):
    binPath: Path to raw/preprocessed .bin data file (AP file from SpikeGLX)
    triggerFilePath: File containing the triggers (each row one value)
    outputPath: filename (including path) of output figure. If None, will use working directory
    surfaceChan: Surface channel (highest channel to include in plot)
    cortexEndChan: Lowest channel to include in plot
    channelRange_toPlot: Range of channel to include in plot. Make sure first channel is lower than second
    windowAroundTrigger: Time (x-axis) to extract aroound triggers in ms
    removeChanOffsets: Removes channel offsets by subtracting median of first 10 seconds for each channel. Not needed for preprocessed data (i.e. CatGT)
    yAxisUM: If True, y axis will show micrometers of depth from surface. If False, y axis will show channel numbers
    plotTriggerTime: If True, will plot a dashed vertical line at the trigger time
'''

# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ------------------ User input ------------------
binPath = Path('F:/example/path')
triggerFilePath = Path('F:/example/path2')
outputPath = None
surfaceChan = 300
channelRange_toPlot = [180, surfaceChan]
windowAroundTrigger = [-2, 5]
removeChanOffsets = False
yAxisUM = True
plotTriggerTime = True
# ------------------------------------------------



# set plot style
plt.rcParams['font.size'] = 15
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.style'] = 'normal'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15


# ------------------ Functions ------------------
def readMeta(binFullPath):
    metaName = binFullPath.stem + ".meta"
    metaPath = Path(binFullPath.parent / metaName)
    metaDict = {}
    if metaPath.exists():
        # print("meta file present")
        with metaPath.open() as f:
            mdatList = f.read().splitlines()
            # convert the list entries into key value pairs
            for m in mdatList:
                csList = m.split(sep='=')
                if csList[0][0] == '~':
                    currKey = csList[0][1:len(csList[0])]
                else:
                    currKey = csList[0]
                metaDict.update({currKey: csList[1]})
    else:
        print("no meta file")
    return(metaDict)

def getTriggers(trigPath, sRate):
    triggers = []
    with trigPath.open() as f:
        triggers_raw = f.read().splitlines()
        for trig in triggers_raw:
            if '.' not in trig:
                triggers.append(int(trig))
            else:
                triggers.append(int(float(trig)*sRate))
    return np.array(triggers)

def getVrange(data):
    mi = np.nanmin(data)
    ma = np.nanmax(data)
    limU = np.max([np.abs(mi), np.abs(ma)])
    limL = -limU
    return limL, limU

def getyticklabels(surfaceChan, channelRange_toPlot, yAxisUM):
    # TODO: Change this to stick to round values
    chans = np.arange(channelRange_toPlot[0],channelRange_toPlot[1])[::-1]
    nChans_toPlot = np.abs(channelRange_toPlot[1] - channelRange_toPlot[0])
    tickSpacing = int(nChans_toPlot / 5)
    yticks = np.arange(nChans_toPlot)
    if yAxisUM:
        chans = (surfaceChan - chans[::-1])*10
    ylabel = 'Depth [µm]' if yAxisUM else 'Channel number'
    return yticks[::tickSpacing], chans[::tickSpacing], ylabel

def getXtickLabels(windowAroundTrigger, sRate):
    # TODO: Change this to stick to round values
    lenWindowMs = np.abs(windowAroundTrigger[1] - windowAroundTrigger[0])
    lenWindowSamples = int(lenWindowMs*1e-3 * sRate)
    xSpacing = int(lenWindowSamples / 5)
    xticks = np.arange(0, lenWindowSamples, xSpacing)
    xticklabels = (xticks/sRate)*1e3 + windowAroundTrigger[0]
    xlabel = 'time after stimulus [ms]'
    return xticks, np.around(xticklabels, decimals=1), xlabel


# -------------------- Load data ------------------------------
# load meta data
meta = readMeta(binPath)
sRate = float(meta['imSampRate'])

# load trigger data
triggers = getTriggers(triggerFilePath, sRate)

# load raw ephys data
nChansSaved = int(meta['nSavedChans'])
nChansEphys = nChansSaved - 1
fileDim = int(int(meta['fileSizeBytes'])/(2*nChansSaved))
rawData = np.memmap(binPath, dtype='int16', mode='r', shape=(nChansSaved, fileDim), offset=0, order='F')
# drop the digital channel
rawData = rawData[:-1, :]

# -------------------- Collect samples -------------------------
# bins in window (in samples)
bins = np.arange(windowAroundTrigger[0]*1e-3*sRate, windowAroundTrigger[1]*1e-3*sRate).astype(int)

snapshot = np.zeros([nChansEphys,len(bins)])
count = 0
for i, trig in enumerate(triggers):
    b = trig + bins
    dat = rawData[:, b].copy()
    snapshot += dat
    count += 1
    del dat
snapshot /= count
# remove channel offsets
if removeChanOffsets:
    for i in range(nChansEphys):
        snapshot[i,:] -= np.median(rawData[i,:int(10*sRate)])
#snapshot = snapshot - np.median(snapshot)

# ------------------------ Plotting ---------------------------------
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10,10))
snapshot -= np.median(snapshot)
vmin, vmax = getVrange(snapshot)
axs[0].imshow(snapshot[channelRange_toPlot[0]:channelRange_toPlot[1],:], aspect = 'auto', origin = 'lower', cmap = plt.cm.RdBu_r, vmin=vmin, vmax=vmax) #plt.cm.RdBu_r

if plotTriggerTime:
    trigTimeSamples = int(-windowAroundTrigger[0]*1e-3*sRate)
    axs[0].axvline(trigTimeSamples, c='k', ls='--', alpha=0.5)
    axs[1].axvline(trigTimeSamples, c='k', ls='--', alpha=0.5)

yticks, yticklabels, ylabel = getyticklabels(surfaceChan, channelRange_toPlot, yAxisUM)
axs[0].set_yticks(yticks)
axs[0].set_yticklabels(yticklabels)
axs[0].set_ylabel(ylabel)

xticks, xticklabels, xlabel = getXtickLabels(windowAroundTrigger, sRate)
axs[0].set_xticks(xticks)
axs[0].set_xticklabels(xticklabels)

# plot the waveform every 5 channels on next plot
channels_toPlot = np.arange(channelRange_toPlot[0], channelRange_toPlot[1], 5)
for i,chan in enumerate(channels_toPlot):
    axs[1].plot(snapshot[chan]+(i*5), c='k')
axs[1].set_ylim([0, len(channels_toPlot)*5])
axs[1].set_xticks(xticks)
axs[1].set_xticklabels(xticklabels)
axs[1].set_yticks([])
# hide axes
axs[1].spines[['top', 'bottom', 'left', 'right']].set_visible(False)

fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlabel("Time from stimulus [ms]")

plt.tight_layout()
if outputPath == None:
    outputPath = Path().absolute().joinpath(f'{binPath.stem}.png')
plt.savefig(outputPath)
plt.show()



