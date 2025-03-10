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
    variance, signals, errors = points[:, 0], points[:, 1], points[:, 2]

    #applies gain to each data point, using A = K_adc(e-/DN)* B for noise, signal, and error
    '''signals = signals * gain
    variance = variance * gain
    errors = errors * gain'''

    #sets log-log plot with error
    plt.errorbar(signals, variance, yerr=errors, fmt='o', label="Shot & Read Noise", capsize=5, capthick=1, elinewidth=1, linestyle = 'dashed')
    slope = ptcm.compute_var_slope(signals, variance)
    print(slope)
    x0, y0 = signals[3], variance[3]

    # Generate extended x-values beyond the data range
    x_min, x_max = 0, max(signals) * 2  # Extend range
    x_extended = np.linspace(x_min, x_max, 100)

    # Compute corresponding y-values using the point-slope equation
    y_extended = slope * (x_extended - x0) + y0

    # Overlay the straight line
    plt.plot(x_extended, y_extended, linestyle="solid", color="purple")
'''Plots Shot noise PTC
points: [[stdDev1, signal1, error1], [stdDev2, signal2, error2], ...]'''
def shotPTC(points, gain):
    variance, signals, errors = points[:, 0], points[:, 1], points[:, 2]

    #applies gain to each data point, using A = K_adc(e-/DN)* B for noise, signal, and error
    '''signals = signals * gain
    variance = variance * gain
    errors = errors * gain'''

    #sets log-log plot with error
    plt.errorbar(signals, variance, yerr=errors, fmt='o', label="Shot Noise", capsize=5, capthick=1, elinewidth=1, linestyle = 'dashdot')

    #compute slope of shot noise line in log-log coordinates
    slope = ptcm.compute_var_slope(signals, variance)
    print(slope)
    plt.text(signals[len(signals)//2], variance[len(variance)//2],
             f"Slope: {slope:.2f}", fontsize=10, color='black')
    x0, y0 = signals[3], variance[3]

    # Generate extended x-values beyond the data range
    x_min, x_max = 0, max(signals) * 2  # Extend range
    x_extended = np.linspace(x_min, x_max, 100)

    # Compute corresponding y-values using the point-slope equation
    y_extended = slope * (x_extended - x0) + y0

    # Overlay the straight line
    plt.plot(x_extended, y_extended, linestyle="solid", color="red")
    
'''Plots total noise PTC
points: [[stdDev1, signal1, error1], [stdDev2, signal2, error2], ...]'''
def totalNoisePTC(points, gain):
    variance, signals, errors = points[:, 0], points[:, 1], points[:, 2]

    #applies gain to each data point, using A = K_adc(e-/DN)* B for noise, signal, and error
    '''signals = signals * gain
    variance = variance * gain
    errors = errors * gain'''
    plt.errorbar(signals, variance, yerr=errors, fmt='o', label="Total Noise", capsize=5, capthick=1, elinewidth=1, linestyle = 'solid')

    plt.xlabel("Signal (DN)")
    plt.ylabel("Noise Variance (DN)")
    '''plt.xlim(0,20)
    plt.ylim(0,2)'''
    plt.xlim(0, signals.max() * 1.1)
    plt.ylim(0, variance.max() * 1.1) # sets y limit to display all points if there is a negative noise value that gets set to 0.01 arbitrarily

def VarPTCGen(offsetImagesFolder, PTCPointsFolder, flatFieldsFolder):
    #extract offset images and average into one, calculate read noise
    offsetImageList = ptcfl.extractFits(offsetImagesFolder)
    offsetImage = np.array(ptcm.sigClippedMeanImage(offsetImageList, 3))
    readVar = np.mean(np.var(offsetImageList, axis = 0))
    print(readVar)

    #extract flatfields and average into one flat field, then find mean signal of the resultant flatfield
    fpnReduced = ptcfl.extractMeanFromCubes(flatFieldsFolder)
    FF_image = fpnReduced[0]
    u_FF = np.mean(FF_image)

    #extract a 4d numpy array, each inner array is an array of images at each illumination level
    PTCImages = np.array(ptcfl.extractFitsCubes(PTCPointsFolder))
    #subtract offset from each image
    PTCData = [ptcm.subtractOffset(img, offsetImage) for img in PTCImages]

    # calculate data for total noise and sort by increasing signal
    totalVarPoints = np.array([ptcm.findVarData(point) for point in PTCData])
    sortedtotalVarPoints = totalVarPoints[np.argsort(totalVarPoints[:, 1])]

    # generate corrected image by pulling fpn out with flatfielding for each image
    noMoreFPN = ptcm.shotAndRead(FF_image, u_FF, np.array(PTCData))
    # calculate total noise from corrected image once fpn is subtracted and sort by increasing signal
    shotAndReadVar = np.array([ptcm.findVarData(point) for point in noMoreFPN])
    sortedShotAndReadVar = shotAndReadVar[np.argsort(shotAndReadVar[:, 1])]
    #calculate shot noise by subtracting read noise from shot+read noise in quadrature
    shotPoints = ptcm.shotVar(sortedShotAndReadVar, readVar)
    #calculate gain for each data point using K_adc(e-/DN) = Signal / (shot noise) ** 2 and average them.
    gains = [shotPoints[i][1]/(shotPoints[i][0]) for i in range(len(shotPoints)) if i > 4 and i < (len(shotPoints) - 2)]
    gain = np.mean(gains)
    #generate PTCs on one log-log plot with error
    totalNoisePTC(sortedtotalVarPoints, gain)
    shotAndReadPTC(sortedShotAndReadVar, gain)
    shotPTC(shotPoints, gain)
    plt.legend()
    plt.show()
#VarPTCGen(r"C:\Users\nocoo\Downloads\PTCwithGain\5_mil_offsets",r"C:\Users\nocoo\Downloads\PTCwithGain\Light",r"C:\Users\nocoo\Downloads\PTCwithGain\3mils")
VarPTCGen("C:\\Users\\nocoo\\OneDrive - University of Arizona\\trialDark", "C:\\Users\\nocoo\\OneDrive - University of Arizona\\trialLight", "C:\\Users\\nocoo\\OneDrive - University of Arizona\\PTCfitsCubes")