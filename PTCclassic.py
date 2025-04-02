'''
This is a program that generates a classical PTC family that is shot noise and FPN dominated. 
It plots an FPN, shot noise, read noise, and total noise PTC on the same log-log plot.
It also calculates the read noise, full well, and Pn (FPN quality factor) of the camera.
'''

import numpy as np
import PTCMathFunctions as ptcm
import matplotlib.pyplot as plt
import importlib
importlib.reload(ptcm)

'''Plots Shot + read noise PTC
ax: axis to plot on
points: [[stdDev1, signal1, error1], [stdDev2, signal2, error2], ...]
sensitivity: K_adc(e-/DN)
'''
def shotAndReadPTC(ax, points, sensitivity):
    stdDevs, signals, errors = points[:, 0], points[:, 1], points[:, 2]

    #applies sensitivity to each data point, using A = K_adc(e-/DN)* B for noise, signal, and error
    signals = signals * sensitivity
    stdDevs = stdDevs * sensitivity
    errors = errors * sensitivity

    #sets log-log plot with error
    ax.errorbar(signals, stdDevs, yerr=errors, fmt='o', label="Shot & Read Noise", capsize=5, capthick=1, elinewidth=1, linestyle = 'dashed')
    ax.set_xscale("log")
    ax.set_yscale("log")

'''Plots Shot noise PTC
ax: axis to plot on
points: [[stdDev1, signal1, error1], [stdDev2, signal2, error2], ...]
sensitivity: K_adc(e-/DN)
'''
def shotPTC(ax, points, sensitivity):
    stdDevs, signals, errors = points[:, 0], points[:, 1], points[:, 2]

    #applies sensitivity to each data point, using A = K_adc(e-/DN)* B for noise, signal, and error
    signals = signals * sensitivity
    stdDevs = stdDevs * sensitivity
    errors = errors * sensitivity

    #sets log-log plot with error
    ax.errorbar(signals, stdDevs, yerr=errors, fmt='o', label="Shot Noise", capsize=5, capthick=1, elinewidth=1, linestyle = 'dashdot')
    ax.set_xscale("log")
    ax.set_yscale("log")

    #compute slope and associated error of shot noise line in log-log coordinates
    slope, err = ptcm.compute_slope(signals, stdDevs)
    ax.text(signals[len(signals)//4], stdDevs[len(stdDevs)//4],
             f"Slope: {slope:.2f} +/- {err:.4f}", fontsize=10, color='black')
'''Plots fpn PTC
ax: axis to plot on
points: [[stdDev1, signal1, error1], [stdDev2, signal2, error2], ...]
sensitivity: K_adc(e-/DN)
'''
def fpnPTC(ax, points, sensitivity):
    stdDevs, signals, errors = points[:, 0], points[:, 1], points[:, 2]

    #applies sensitivity to each data point, using A = K_adc(e-/DN)* B for noise, signal, and error
    signals = signals * sensitivity
    stdDevs = stdDevs * sensitivity
    errors = errors * sensitivity

    #sets log-log plot with error
    ax.errorbar(signals, stdDevs, yerr=errors, fmt='o', label="fpn", capsize=5, capthick=1, elinewidth=1, linestyle = 'dotted')
    ax.set_xscale("log")
    ax.set_yscale("log")

    #compute slope and associated error of fpn line in log-log coordinates
    slope, err = ptcm.compute_slope(signals, stdDevs)

    ax.text(signals[len(signals)//2], stdDevs[len(stdDevs)//2],
             f"Slope: {slope:.2f} +/- {err:.4f}", fontsize=10, color='black')
    
'''Plots total noise PTC
ax: axis to plot on
points: [[stdDev1, signal1, error1], [stdDev2, signal2, error2], ...]
sensitivity: K_adc(e-/DN)
'''
def totalNoisePTC(ax, points, sensitivity):
    stdDevs, signals, errors = points[:, 0], points[:, 1], points[:, 2]
    #applies sensitivity to each data point, using A = K_adc(e-/DN)* B for noise, signal, and error
    signals = signals * sensitivity
    stdDevs = stdDevs * sensitivity
    errors = errors * sensitivity
    ax.errorbar(signals, stdDevs, yerr=errors, fmt='o', label="Total Noise", capsize=5, capthick=1, elinewidth=1, linestyle = 'solid')
    
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Signal (e-)")
    ax.set_ylabel("Noise (e-)")
    ax.set_xlim(signals.min() * 0.9, signals.max() * 1.1)
    ax.set_ylim(0.01 * sensitivity, stdDevs.max() * 1.1) # sets y limit to display all points if there is a negative noise value that gets set to 0.01 arbitrarily

'''
runs classic PTC generation, plots in logspace in e- units
offsetImagesList: an array of 2d fits images for offset calculation
PTCImages: an array of 3d fits cubes, one for each illumination level
fpnReduced: an array of 3d fits cubes for flatfielding flatfield 
sensitivity: K_adc(e-/DN)
fig: the figure to plot on
'''
def PTCGen(offsetImageList, PTCImages, fpnReduced, sensitivity, fig):
    # average offset images into one, calculate read noise
    offsetImage = np.array(ptcm.sigClippedMeanImage(offsetImageList, 3))
    readNoise = np.mean(np.std(offsetImageList, axis = 0))
    readNoise_err = np.std(np.std(offsetImageList, axis = 0)) 

    # extract flatfields and average into one flat field, then find mean signal of the resultant flatfield
    FF_image = ptcm.subtractOffset(fpnReduced[0], offsetImage)
    u_FF = np.mean(FF_image)

    #subtract offset from each image
    PTCData = [ptcm.subtractOffset(img, offsetImage) for img in PTCImages]

    # calculate data for total noise and sort by increasing signal
    totalNoisePoints = np.array([ptcm.findData(point) for point in PTCData])
    sortedtotalNoisePoints = totalNoisePoints[np.argsort(totalNoisePoints[:, 1])]
    # generate corrected image by pulling fpn out with flatfielding for each image
    noMoreFPN = ptcm.shotAndRead(FF_image, u_FF, np.array(PTCData))
    # calculate total noise from corrected image once fpn is subtracted and sort by increasing signal
    shotAndReadNoise = np.array([ptcm.findData(point) for point in noMoreFPN])
    sortedShotAndReadNoise = shotAndReadNoise[np.argsort(shotAndReadNoise[:, 1])]

    # calculate fpn by subtracting shot+read noise from total noise in quadrature
    fpnPoints = ptcm.fpn(sortedShotAndReadNoise, sortedtotalNoisePoints)
    #calculate shot noise by subtracting read noise from shot+read noise in quadrature
    shotPoints = ptcm.shotNoise(sortedShotAndReadNoise, readNoise)
    fullWell, fullWell_err =  ptcm.nonlinearityPoint(fpnPoints[:, 1], fpnPoints[:, 0])
    #calculate FPN quality factor for each data point using Pn = FPN / Signal and average them.
    Pns = [fpnPoints[i][0]/fpnPoints[i][1] for i in range(len(fpnPoints)) if i > 6 and i < (len(fpnPoints) - 4)]
    np.clip(Pns, np.mean(Pns) - 2 * np.std(Pns), np.mean(Pns) + 2 * np.std(Pns))
    Pn_err = np.std(Pns)
    Pn = np.mean(Pns)
    #generate PTCs for each noise on one log-log plot with error
    ax = fig.axes[1]
    totalNoisePTC(ax, sortedtotalNoisePoints, sensitivity)
    shotAndReadPTC(ax, sortedShotAndReadNoise, sensitivity)
    shotPTC(ax, shotPoints, sensitivity)
    fpnPTC(ax, fpnPoints, sensitivity)
    #add read noise line
    ax.axhline(readNoise * sensitivity, linestyle="--", color="purple", alpha=0.7, label="Read Noise")
    ax.legend()
    fig.canvas.draw()
    return (readNoise, readNoise_err), (Pn, Pn_err), (fullWell, fullWell_err)
