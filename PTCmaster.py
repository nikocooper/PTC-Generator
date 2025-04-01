'''
This script generates the PTC plots and calculates camera characterization parameters.
It formats all plots on one matplotlib figure, and displays the results in a text box.
'''
import PTCclassic as PTCC
import PTCvariance as PTCV
import PTCSignaltoNoise as PTCSN
import PTCMathFunctions as PTCM
import PTCFileLoader as PTCFL
import matplotlib.pyplot as plt
import numpy as np
# Reload modules to ensure they are up to date
import importlib
importlib.reload(PTCM)
importlib.reload(PTCC)
importlib.reload(PTCV)
importlib.reload(PTCSN)
importlib.reload(PTCFL)

'''This function generates the PTC plots and calculates camera characterization parameters.
It takes in three folders:
1. offsetImagesFolder: folder containing a 3d fits cube for offset calculation
2. PTCPointsFolder: folder containing 3d fits cubes for each illumination level
3. flatFieldsFolder: folder containing 3d fits cubes for flat field correction 
'''
def generator(offsetImagesFolder, PTCPointsFolder, flatFieldsFolder):

    # Extract data from the input folders
    offsetImageList = PTCFL.extractFitsCubes(offsetImagesFolder)[0]
    fpnReduced = PTCFL.extractMeanFromCubes(flatFieldsFolder)
    PTCImages = np.array(PTCFL.extractFitsCubes(PTCPointsFolder))

    # Generate PTCs and extract data
    fig, _ = plt.subplots(2, 2, figsize=(14, 12))

    sensitivity = PTCV.VarPTCGen(offsetImageList, PTCImages, fpnReduced, fig)
    readNoise, Pn, fullWell = PTCC.PTCGen(offsetImageList, PTCImages, fpnReduced, sensitivity, fig)
    StoNlim = PTCSN.SvNPTCGen(offsetImageList, PTCImages, fpnReduced, sensitivity, fig)

    ax1, ax2, ax3, ax_text = fig.axes[0], fig.axes[1], fig.axes[2], fig.axes[3]
    # Add titles to each subplot
    ax1.set_title("Variance PTC")
    ax2.set_title("Classic PTC")
    ax3.set_title("Signal to Noise PTC")

    ax_text.axis('off')  # Hide axis, as we're only displaying text

    # Add extracted values to the text figure
    ax_text.text(0.5, 0.75, 'Camera Characterization Results', ha='center', va='center', fontsize=12, fontweight='bold')
    ax_text.text(0.5, 0.65, 'Minimum Detectable Signal (e-): ' + str((1+np.sqrt(1+4*(readNoise*sensitivity)**2))/2), ha='center', va='center', fontsize=8)
    ax_text.text(0.5, 0.55, 'Signal to Noise Limit (dB): ' + str(20*np.log10(StoNlim)), ha='center', va='center', fontsize=8)
    ax_text.text(0.5, 0.45, 'Dynamic Range (dB): ' + str(20*np.log10(fullWell/readNoise)), ha='center', va='center', fontsize=8)
    ax_text.text(0.5, 0.35, 'Full Well (e-): ' + str(fullWell * sensitivity), ha='center', va='center', fontsize=8)
    ax_text.text(0.5, 0.25, "Sensitivity (e-/DN): " + str(sensitivity), ha='center', va='center', fontsize=8)
    ax_text.text(0.5, 0.15, "Pn: " + str(round(Pn*100,2))+"%", ha='center', va='center', fontsize=8)
    ax_text.text(0.5, 0.05, 'Read Noise (e-): ' + str(readNoise * sensitivity), ha='center', va='center', fontsize=8)

    fig.canvas.draw()
    # Show the combined figure
    plt.tight_layout()
    fig.subplots_adjust(wspace=0.3, hspace=0.3, bottom=0.15)
    plt.show()

generator(<path_to_offsets>, <path_to_illuminated_images>, <path_to_flats>)
