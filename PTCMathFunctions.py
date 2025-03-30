import numpy as np

#averages a list of images into one mean image pixel by pixel
def meanImage(imageList):
    return np.mean(imageList, axis=0)

def stdDev(image):
    return np.std(image)

def findSignal(image):
    mean = np.mean(image)
    return mean

#std deviation of image with sigma clipping
def sigClippedStdDev(image, sig):
    mean = np.mean(image)
    std = stdDev(image)
    return stdDev(np.clip(image, mean-sig*std, mean+sig*std))

#mean of the image with sigma clipping
def findSigClippedSignal(image, sig):
    mean = np.mean(image)
    std = stdDev(image)
    return np.mean(np.clip(image, mean-sig*std, mean+sig*std))

'''Used functions for PTC generation'''

#subtracts averaged offset from raw image (pixel by pixel subtraction)
def subtractOffset(image, offsetImage): 
    return image - offsetImage 

#averages a list of images into one mean image pixel by pixel with sigma clipping
def sigClippedMeanImage(imageList, sig):
    meanIm = meanImage(imageList)
    mean = findSigClippedSignal(meanIm, sig)
    std = sigClippedStdDev(meanIm, sig)
    clippedImageList = [np.clip(image, mean-sig*std, mean+sig*std) for image in imageList]
    return np.mean(clippedImageList, axis=0)


def findData(imageList):
    imageArray = np.array(imageList)  # Convert to NumPy array
    
    # Check which images have both pixels < 50 and > 250 (removes image with bright whte strip)
    mask_invalid = np.any(imageArray < 100, axis=(1, 2)) & np.any(imageArray > 250, axis=(1, 2))
    valid_images = imageArray[~mask_invalid]
    #catch so program doesn't fail if a stack of images all has the white strip
    if valid_images.size == 0:
        return [1, 1, 1]

    mean_signals = np.mean(valid_images, axis=(1, 2))  # List with signal for each image
    std_devs = np.std(valid_images, axis=(1, 2))      # List with noise(std dev) for each image
    
    #average the signal and noise for the illumination level and find error in the noise
    avg_mean_signal = np.mean(mean_signals)
    avg_std_dev = np.mean(std_devs)
    error = np.std(std_devs)
    
    return [avg_std_dev, avg_mean_signal, error]

def findSvNData(imageList):
    imageArray = np.array(imageList)  # Convert to NumPy array
    
    # Check which images have both pixels < 50 and > 250 (removes image with bright whte strip)
    mask_invalid = np.any(imageArray < 100, axis=(1, 2)) & np.any(imageArray > 250, axis=(1, 2))
    valid_images = imageArray[~mask_invalid]
    #catch so program doesn't fail if a stack of images all has the white strip
    if valid_images.size == 0:
        return [1, 1, 1]

    mean_signals = np.mean(valid_images, axis=(1, 2))  # List with signal for each image
    std_devs = np.std(valid_images, axis=(1, 2))      # List with noise(std dev) for each image
    SvNs = np.divide(mean_signals, std_devs)  # Calculate signal to noise ratio
    #average the signal and noise for the illumination level and find error in the noise
    avg_mean_signal = np.mean(mean_signals)
    avg_SvN = np.mean(SvNs)  # Calculate signal to noise ratio
    error = np.std(SvNs)
    
    return [avg_SvN, avg_mean_signal, error]

def findVarData(imageList):
    imageArray = np.array(imageList)  # Convert to NumPy array
    
    # Check which images have both pixels < 50 and > 250 (removes image with bright whte strip)
    mask_invalid = np.any(imageArray < 100, axis=(1, 2)) & np.any(imageArray > 250, axis=(1, 2))
    valid_images = imageArray[~mask_invalid]
    #catch so program doesn't fail if a stack of images all has the white strip
    if valid_images.size == 0:
        return [1, 1, 1]

    mean_signals = np.mean(valid_images, axis=(1, 2))  # List with signal for each image
    variance = np.var(valid_images, axis=(1, 2))      # List with noise(std dev) for each image
    
    #average the signal and noise for the illumination level and find error in the noise
    avg_mean_signal = np.mean(mean_signals)
    avg_variance = np.mean(variance)
    error = np.std(variance)
    
    return [avg_variance, avg_mean_signal, error]

''' Calculate shot noise for each illumination level'
shotAndReadNoiseArray: [[stdDev1, signal1, error1], [stdDev2, signal2, error2], ...]
readNoise: float'''
def shotVar(shotAndReadNoiseArray, readNoise): 
    #Extract noise and error
    shotAndReadNoise = shotAndReadNoiseArray[:, 0]
    errors_SR = shotAndReadNoiseArray[:, 2]

    #Compute shot noise using N_shot = sqrt(N_shot+read ** 2 - N_read ** 2)
    diff = shotAndReadNoise - readNoise
    diff = np.maximum(diff, 0.001)  # Prevent negative values
    shotNoise = diff

    # Error propagation formula
    errors_S = np.abs(shotAndReadNoise / shotNoise) * errors_SR  

    # Create output array with propagated errors
    shotNoiseArray = shotAndReadNoiseArray.copy()
    shotNoiseArray[:, 0] = shotNoise
    shotNoiseArray[:, 2] = errors_S 
    return shotNoiseArray

''' Calculate shot noise for each illumination level'
shotAndReadNoiseArray: [[stdDev1, signal1, error1], [stdDev2, signal2, error2], ...]
readNoise: float'''
def shotNoise(shotAndReadNoiseArray, readNoise): 
    #Extract noise and error
    shotAndReadNoise = shotAndReadNoiseArray[:, 0]
    errors_SR = shotAndReadNoiseArray[:, 2]

    #Compute shot noise using N_shot = sqrt(N_shot+read ** 2 - N_read ** 2)
    diff = shotAndReadNoise ** 2 - readNoise ** 2
    diff = np.maximum(diff, 0.001)  # Prevent negative values
    shotNoise = np.sqrt(diff)

    # Error propagation formula
    errors_S = np.abs(shotAndReadNoise / shotNoise) * errors_SR  

    # Create output array with propagated errors
    shotNoiseArray = shotAndReadNoiseArray.copy()
    shotNoiseArray[:, 0] = shotNoise
    shotNoiseArray[:, 2] = errors_S 

    return shotNoiseArray
''' Calculate fpn for each illumination level'
shotAndReadNoiseArray: [[stdDev1, signal1, error1], [stdDev2, signal2, error2], ...]
totalNoiseArray: [[stdDev1, signal1, error1], [stdDev2, signal2, error2], ...]'''
def fpn(shotAndReadNoiseArray, totalNoiseArray):
    #Extract noise and error
    shotAndReadNoise = shotAndReadNoiseArray[:, 0]
    errors_SR = shotAndReadNoiseArray[:, 2]
    totalNoise = totalNoiseArray[:, 0]
    errors_T = totalNoiseArray[:, 2]

    #Compute fpn using N_fpn = sqrt(N_total ** 2 - N_shot+read ** 2)
    diff = totalNoise ** 2 - shotAndReadNoise ** 2
    diff = np.maximum(diff, 0.001)  # Prevent negative values
    fpn = np.sqrt(diff)

    # Error propagation formula
    errors_F = np.sqrt(
        (np.abs(totalNoise / fpn) * errors_T) ** 2 +
        (np.abs(shotAndReadNoise / fpn) * errors_SR) ** 2
    )

    # Create output array with propagated errors
    fpnArray = shotAndReadNoiseArray.copy()
    fpnArray[:, 0] = fpn
    fpnArray[:, 2] = errors_F  

    return fpnArray

''' Calculate fpn for each illumination level via flatfielding'
FF_image: 2d numpy array
u_FF: mean signal of FF_image
imArray: 4d numpy array containing each illuminated image in the PTC after offset subtraction'''
def shotAndRead(FF_Image, u_FF, imArray):
    corr = u_FF*(imArray / FF_Image)
    return corr

'''returns quantum efficiency of a camera in e-/photon'
Luminance: W/m^2, Signal: e-, Exposure: s, Wavelength: m, Area: m^2'''
def QuantumEff(Luminance, Signals, Exposures, Wavelength, Area):
    #Calculate incident photon flux
    fluxes = (Luminance * Exposures * Area)/ (((6.62607015 * 10 ** -34) * (3 * 10 ** 8)) / Wavelength)
    #Calculate quantum efficiency
    qes = [(Signals[i]) / fluxes[i] for i in range(len(fluxes)) if i > 4 and i < (len(fluxes) - 3)]
    print("qes: ", qes)
    return np.mean(qes)
#Used for curve fitting
def linear_func(log_x, slope, intercept):
        return slope * log_x + intercept
#calculates slope of lines in a log-log plot
def compute_slope(x, y):
    log_x = np.log10(x)
    log_y = np.log10(y)
    start = len(log_x) // 2
    end = len(log_x) - 4

    # Compute slope
    slope = (log_y[end] - log_y[start]) / (log_x[end] - log_x[start])

    return slope
#calculates slop of lines in a log-log plot ignoring saturated images
def compute_var_slope(x, y):
    start = len(x)//2
    end = len(x) - 4
    slope = (y[end] - y[start]) / (x[end] - x[start])
    
    return slope

def nonlinearityPoint(x, y):
    """
    Finds the x-value just before the y-values drop and do not recover.

    Args:
        x (array-like): X data values.
        y (array-like): Y data values.

    Returns:
        float: X-value before the drop.
    """
    y = np.array(y)
    
    # Iterate to find the first major drop
    for i in range(1, len(y)):
        if y[i] < y[i - 1]:  # Drop detected
            # If this is the last point, return the previous x-value
            if i == len(y) - 1:
                return x[i - 1]
            
            # Check if the drop is sustained
            if all(y[j] <= y[i] for j in range(i, len(y))):
                return x[i - 1]  # Return the last stable x-value before the drop

    return x[-1]  # If no drop found, return the last x-value
def maxStoN(x, y):
    log_x = np.log10(x)
    log_y = np.log10(y)
    
    slope = (log_y[-4] - log_y[6]) / (log_x[-4] - log_x[6])
    for i in range(len(log_x)//2, len(log_x) - 1):
        new_slope = (log_y[i] - log_y[i-1]) / (log_x[i] - log_x[i-1])
        if abs(new_slope - slope) > slope * 0.5:
            return y[i-1]
    print("maxStoN: ", y[-1])
    return y[-1]