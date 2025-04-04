'''This program generates a variance PTC for a given set of images.
It plots the total variance PTC, the shot variance PTC, and the shot + read variance PTC.
It calculates the sensitivity of the camera.'''
import numpy as np
import PTCMathFunctions as ptcm
import matplotlib.pyplot as plt
import importlib
importlib.reload(ptcm)

'''Plots Shot + read noise variance PTC
ax: axis to plot on
points: [[stdDev1, signal1, error1], [stdDev2, signal2, error2], ...]
'''
def shotAndReadPTC(ax,points):
    variance, signals, errors = points[:, 0], points[:, 1], points[:, 2]


    #sets plot with error
    ax.errorbar(signals, variance, yerr=errors, fmt='o', label="Shot & Read Noise Variance", capsize=5, capthick=1, elinewidth=1, linestyle = 'dashed')

'''Plots Shot noise variance PTC
ax: axis to plot on
points: [[stdDev1, signal1, error1], [stdDev2, signal2, error2], ...]
'''
def shotPTC(ax, points):
    global slope
    variance, signals, errors = points[:, 0], points[:, 1], points[:, 2]


    #sets plot with error
    ax.errorbar(signals, variance, yerr=errors, fmt='o', label="Shot Noise Variance", capsize=5, capthick=1, elinewidth=1, linestyle = 'dashdot')

    #compute slope and associated error of shot noise line
    slope, err = ptcm.compute_var_slope(signals, variance)
    ax.text(signals[len(signals)//2], variance[len(variance)//2],
             f"Slope: {slope:.2f} +/- {err:.4f}", fontsize=10, color='black')
    return 1/slope, err/(slope ** 2)

'''Plots total noise variance PTC
ax: axis to plot on
points: [[stdDev1, signal1, error1], [stdDev2, signal2, error2], ...]
'''
def totalNoisePTC(ax, points):
    variance, signals, errors = points[:, 0], points[:, 1], points[:, 2]

    ax.errorbar(signals, variance, yerr=errors, fmt='o', label="Total Noise Variance", capsize=5, capthick=1, elinewidth=1, linestyle = 'solid')

    ax.set_xlabel("Signal (DN)")
    ax.set_ylabel("Noise Variance (DN)")
    ax.set_xlim(0, signals.max() * 1.1)
    ax.set_ylim(0, variance.max() * 1.1) 

'''
runs variance PTC generation, plots on standard x-y coordinates in DN units
offsetImagesList: an array of 2d fits images for offset calculation
PTCImages: an array of 3d fits cubes, one for each illumination level
fpnReduced: an array of 3d fits cubes for flatfielding flatfield 
sensitivity: K_adc(e-/DN)
fig: the figure to plot on
'''
def VarPTCGen(offsetImageList, PTCImages, fpnReduced, fig):
    # average offset images into one, calculate read noise
    offsetImage = np.array(ptcm.sigClippedMeanImage(offsetImageList, 3))
    #calculate read noise variance per image for averaging and error calculation
    readVar = np.mean(np.var(offsetImageList, axis = (1,2)))
    readVar_err = np.std(np.var(offsetImageList, axis = (1,2))) 

    # extract flatfields and average into one flat field, then find mean signal of the resultant flatfield
    FF_image = ptcm.subtractOffset(fpnReduced[0], offsetImage)
    u_FF = np.mean(FF_image)

    #subtract offset from each image
    PTCData = [ptcm.subtractOffset(img, offsetImage) for img in PTCImages]

    # calculate data for total noise variance and sort by increasing signal
    totalVarPoints = np.array([ptcm.findVarData(point) for point in PTCData])
    sortedtotalVarPoints = totalVarPoints[np.argsort(totalVarPoints[:, 1])]

    # generate corrected image by pulling fpn out with flatfielding for each image
    noMoreFPN = ptcm.shotAndRead(FF_image, u_FF, np.array(PTCData))
    # calculate shot+read noise variance from corrected image once fpn is subtracted and sort by increasing signal
    shotAndReadVar = np.array([ptcm.findVarData(point) for point in noMoreFPN])
    sortedShotAndReadVar = shotAndReadVar[np.argsort(shotAndReadVar[:, 1])]
    #calculate shot noise variance by subtracting read noise from shot+read noise in quadrature
    shotPoints = ptcm.shotVar(sortedShotAndReadVar, readVar, readVar_err)
    #generate PTCs on one plot with error
    ax = fig.axes[0]
    totalNoisePTC(ax, sortedtotalVarPoints)
    shotAndReadPTC(ax, sortedShotAndReadVar)

    #compute sensitivity as the inverse of the slope of the shot noise line
    sens, err = shotPTC(ax, shotPoints)
    ax.legend()
    fig.canvas.draw()
    return (sens, err)
