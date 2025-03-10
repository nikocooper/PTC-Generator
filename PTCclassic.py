'''
This is a program that generates a classical PTC family that is shot noise and FPN dominated. 
Currently, calculates gain from each good data point on the shot noise graph and averages the gain,
then multiplies each signal, noise, and error point by the gain to achieve a PTC in e- units. Gain here is
system gain (K_adc(e-/DN)), which is actually called sensitivity. There appears to be something wrong in
the logic with which I am calculating and applying gain. For the data set where gain is set to 0, I am calculating
46.95, and for the data set where gain is set to 6, I am calculating 22.85. Gain is applied in the fpnPTC, shotPTC,
shotAndReadPTC, and totalNoisePTC functions. You can comment out the gain application to get a PTC in DN units.
Note that the 0 gain PTC is irregular at the bottom because I didn't delete the corrupted offset images.
Access the code by calling PTCGen(offsetImageFolderPath, IlluminatedImageFolderPath, flatFieldFolderPath).'''
import numpy as np
import PTCMathFunctions as ptcm
import PTCFileLoader as ptcfl
import matplotlib.pyplot as plt
import importlib
importlib.reload(ptcm)
importlib.reload(ptcfl)

'''Plots Shot + read noise PTC
points: [[stdDev1, signal1, error1], [stdDev2, signal2, error2], ...]'''
def shotAndReadPTC(points, gain):
    stdDevs, signals, errors = points[:, 0], points[:, 1], points[:, 2]

    #applies gain to each data point, using A = K_adc(e-/DN)* B for noise, signal, and error
    signals = signals * gain
    stdDevs = stdDevs * gain
    errors = errors * gain

    #sets log-log plot with error
    plt.errorbar(signals, stdDevs, yerr=errors, fmt='o', label="Shot & Read Noise", capsize=5, capthick=1, elinewidth=1, linestyle = 'dashed')
    plt.xscale("log")
    plt.yscale("log")

'''Plots Shot noise PTC
points: [[stdDev1, signal1, error1], [stdDev2, signal2, error2], ...]'''
def shotPTC(points, gain):
    stdDevs, signals, errors = points[:, 0], points[:, 1], points[:, 2]

    #applies gain to each data point, using A = K_adc(e-/DN)* B for noise, signal, and error
    signals = signals * gain
    stdDevs = stdDevs * gain
    errors = errors * gain

    #sets log-log plot with error
    plt.errorbar(signals, stdDevs, yerr=errors, fmt='o', label="Shot Noise", capsize=5, capthick=1, elinewidth=1, linestyle = 'dashdot')
    plt.xscale("log")
    plt.yscale("log")

    #compute slope of shot noise line in log-log coordinates
    slope = ptcm.compute_slope(signals, stdDevs)
    plt.text(signals[len(signals)//4], stdDevs[len(stdDevs)//4],
             f"Slope: {slope:.2f}", fontsize=10, color='black')

'''Plots fpn PTC
points: [[stdDev1, signal1, error1], [stdDev2, signal2, error2], ...]'''
def fpnPTC(points, gain):
    stdDevs, signals, errors = points[:, 0], points[:, 1], points[:, 2]

    #applies gain to each data point, using A = K_adc(e-/DN)* B for noise, signal, and error
    signals = signals * gain
    stdDevs = stdDevs * gain
    errors = errors * gain

    #sets log-log plot with error
    plt.errorbar(signals, stdDevs, yerr=errors, fmt='o', label="fpn", capsize=5, capthick=1, elinewidth=1, linestyle = 'dotted')
    plt.xscale("log")
    plt.yscale("log")

    #compute slope of fpn line in log-log coordinates
    slope = ptcm.compute_slope(signals, stdDevs)
    plt.text(signals[len(signals)//2], stdDevs[len(stdDevs)//2],
             f"Slope: {slope:.2f}", fontsize=10, color='black')

'''Plots total noise PTC
points: [[stdDev1, signal1, error1], [stdDev2, signal2, error2], ...]'''
def totalNoisePTC(points, gain):
    stdDevs, signals, errors = points[:, 0], points[:, 1], points[:, 2]

    #applies gain to each data point, using A = K_adc(e-/DN)* B for noise, signal, and error
    signals = signals * gain
    stdDevs = stdDevs * gain
    errors = errors * gain
    plt.errorbar(signals, stdDevs, yerr=errors, fmt='o', label="Total Noise", capsize=5, capthick=1, elinewidth=1, linestyle = 'solid')

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Signal (e-)")
    plt.ylabel("Noise (e-)")
    plt.xlim(signals.min() * 0.9, signals.max() * 1.1)
    plt.ylim(0.01 * gain, stdDevs.max() * 1.1) # sets y limit to display all points if there is a negative noise value that gets set to 0.01 arbitrarily

'''runs PTC generation, this is what you call to start the program
offsetImagesFile: folder of 2d fits images
PTCPointsFolder: folder of 3d fits cubes, one for each illumination level
flatFieldsFolder: folder of 3d fits cubes, each a flatfield at different illumintations'''
def PTCGen(offsetImagesFolder, PTCPointsFolder, flatFieldsFolder):
    #extract offset images and average into one, calculate read noise
    offsetImageList = ptcfl.extractFits(offsetImagesFolder)
    offsetImage = np.array(ptcm.sigClippedMeanImage(offsetImageList, 3))
    readNoise = np.mean(np.std(offsetImageList, axis = 0))
    print(readNoise)

    #extract flatfields and average into one flat field, then find mean signal of the resultant flatfield
    fpnReduced = ptcfl.extractMeanFromCubes(flatFieldsFolder)
    FF_image = fpnReduced[0]
    u_FF = np.mean(FF_image)

    #extract a 4d numpy array, each inner array is an array of images at each illumination level
    PTCImages = np.array(ptcfl.extractFitsCubes(PTCPointsFolder))
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
    print(shotPoints)
    #calculate gain for each data point using K_adc(e-/DN) = Signal / (shot noise) ** 2 and average them.
    gains = [shotPoints[i][1]/(shotPoints[i][0]**2) for i in range(len(shotPoints)) if i > 4 and i < (len(shotPoints) - 2)]
    gain = np.mean(gains)
    print(gain)
    #generate PTCs on one log-log plot with error
    totalNoisePTC(sortedtotalNoisePoints, gain)
    shotAndReadPTC(sortedShotAndReadNoise, gain)
    shotPTC(shotPoints, gain)
    fpnPTC(fpnPoints, gain)
    plt.legend()
    plt.show()

#These are the file locations in my computer
#PTCGen("C:\\Users\\nocoo\\OneDrive - University of Arizona\\trialDark", "C:\\Users\\nocoo\\OneDrive - University of Arizona\\trialLight", "C:\\Users\\nocoo\\OneDrive - University of Arizona\\PTCfitsCubes")
PTCGen(r"C:\Users\nocoo\Downloads\PTCwithGain\5_mil_offsets",r"C:\Users\nocoo\Downloads\PTCwithGain\Light",r"C:\Users\nocoo\Downloads\PTCwithGain\3mils")

