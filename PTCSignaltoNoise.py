'''This program generates a signal to noise PTC for a given set of images.
It plots the total signal to noise ratio, and the signal to nlise ratio with 
fpn subtracted. It calculates the maximum SNR with fpn subtracted.'''
import numpy as np
import PTCMathFunctions as ptcm
import matplotlib.pyplot as plt
import importlib
importlib.reload(ptcm)

'''Plots signal to shot + read noise PTC
ax: axis to plot on
points: [[SvN1, signal1, error1], [SvN2, signal2, error2], ...]
sensitivity: K_adc(e-/DN)
'''
def shotAndReadPTC(ax, points, sensitivity):
    SvNs, signals, errors = points[:, 0], points[:, 1], points[:, 2]

    #convert signal to e- units
    signals = signals * sensitivity
    #sets log-log plot with error
    ax.errorbar(signals, SvNs, yerr=errors, fmt='o', label="Signal to Shot & Read Noise", capsize=5, capthick=1, elinewidth=1, linestyle = 'dashed')
    ax.set_xscale("log")
    ax.set_yscale("log")


'''Plots signal to total noise PTC
ax: axis to plot on
points: [[SvN1, signal1, error1], [SvN2, signal2, error2], ...]
sensitivity: K_adc(e-/DN)
'''
def totalNoisePTC(ax, points, sensitivity):
    SvNs, signals, errors = points[:, 0], points[:, 1], points[:, 2]

    #convert signal to e- units
    signals = signals * sensitivity
    ax.errorbar(signals, SvNs, yerr=errors, fmt='o', label=" Signal to Total Noise", capsize=5, capthick=1, elinewidth=1, linestyle = 'solid')

    #sets axes and labels
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Signal (e-)")
    ax.set_ylabel("Signal to Noise")
    ax.set_xlim(signals.min() * 0.9, signals.max() * 1.1)
    ax.set_ylim(SvNs.min() * 0.9, 150) 

'''
runs signal to noise PTC generation, plots in logspace in e- units
offsetImagesList: an array of 2d fits images for offset calculation
PTCImages: an array of 3d fits cubes, one for each illumination level
fpnReduced: an array of 3d fits cubes for flatfielding flatfield 
sensitivity: K_adc(e-/DN)
fig: the figure to plot on
'''
def SvNPTCGen(offsetImageList, PTCImages, fpnReduced, sensitivity, fig):
    # average offset images into one
    offsetImage = np.array(ptcm.sigClippedMeanImage(offsetImageList, 3))

    # extract flatfields and average into one flat field, then find mean signal of the resultant flatfield
    FF_image = ptcm.subtractOffset(fpnReduced[0], offsetImage)
    u_FF = np.mean(FF_image)

    #subtract offset from each image
    PTCData = [ptcm.subtractOffset(img, offsetImage) for img in PTCImages]

    # calculate data for signal vs total noise and sort by increasing signal
    totalNoisePoints = np.array([ptcm.findSvNData(point) for point in PTCData])
    sortedtotalNoisePoints = totalNoisePoints[np.argsort(totalNoisePoints[:, 1])]
    # generate corrected image by pulling fpn out with flatfielding for each image
    noMoreFPN = ptcm.shotAndRead(FF_image, u_FF, np.array(PTCData))
    # calculate signal vs noise from corrected image once fpn is subtracted and sort by increasing signal
    shotAndReadNoise = np.array([ptcm.findSvNData(point) for point in noMoreFPN])
    sortedShotAndReadNoise = shotAndReadNoise[np.argsort(shotAndReadNoise[:, 1])]
    StoNlim, err = ptcm.maxStoN(sortedShotAndReadNoise[:, 1], sortedShotAndReadNoise[:, 0])
    #generate PTCs on one log-log plot with error
    ax = fig.axes[2]
    totalNoisePTC(ax, sortedtotalNoisePoints, sensitivity)
    shotAndReadPTC(ax, sortedShotAndReadNoise, sensitivity)
    ax.legend()
    fig.canvas.draw()
    return (StoNlim, err)