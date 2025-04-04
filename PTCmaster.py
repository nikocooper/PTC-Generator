'''
Author: Nikoli Cooper
Date: 4/4/2025
This script generates the PTC plots and calculates camera characterization parameters.
A subarray is selected to reduce computation time. This subarray can be the entire array.
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
4. rowIndex: tuple indicating the row range to analyze (start, end)
5. colIndex: tuple indicating the column range to analyze (start, end)
'''
def generator(offsetImagesFolder, PTCPointsFolder, flatFieldsFolder, rowIndex, colIndex):

    # Extract data from the input folders
    offsetImageList = (PTCFL.extractFitsCubes(offsetImagesFolder)[0])[:, rowIndex[0]:rowIndex[1], colIndex[0]:colIndex[1]]
    fpnReduced = (PTCFL.extractMeanFromCubes(flatFieldsFolder))[:, rowIndex[0]:rowIndex[1], colIndex[0]:colIndex[1]]
    PTCImages = (PTCFL.extractFitsCubes(PTCPointsFolder))[:, :, rowIndex[0]:rowIndex[1], colIndex[0]:colIndex[1]]

    # create figure for plotting PTCs
    fig, _ = plt.subplots(2, 2, figsize=(14, 12))

    # Generate PTCs and extract data
    (sensitivity, sensErr) = PTCV.VarPTCGen(offsetImageList, PTCImages, fpnReduced, fig)
    (readNoise, readNoise_err), (Pn, Pn_err), (fullWell, fullWell_err) = PTCC.PTCGen(offsetImageList, PTCImages, fpnReduced, sensitivity, fig)
    (StoNlim, svn_err) = PTCSN.SvNPTCGen(offsetImageList, PTCImages, fpnReduced, sensitivity, fig)

    ax1, ax2, ax3, ax_text = fig.axes[0], fig.axes[1], fig.axes[2], fig.axes[3]
    # Add titles to each subplot
    ax1.set_title("Variance PTC")
    ax2.set_title("Classic PTC")
    ax3.set_title("Signal to Noise PTC")

    ax_text.axis('off')  # Hide axis, as we're only displaying text

    # Add extracted values to the text figure with propogated errors
    ax_text.text(0.5, 0.75, 'Camera Characterization Results', ha='center', va='center', fontsize=12, fontweight='bold')

    ax_text.text(0.5, 0.65, 'Minimum Detectable Signal (e-): ' + "{:.2f}".format(round((1+np.sqrt(1+4*(readNoise*sensitivity)**2))/2,2)) + " +/- " + \
        "{:.3f}".format(round( ((2 * readNoise * sensitivity) / np.sqrt(1 + 4 * (readNoise * sensitivity)**2)) * np.sqrt( \
            (readNoise_err * sensitivity)**2 + (sensErr * readNoise)**2), 3)), ha='center', va='center', fontsize=8)
    
    ax_text.text(0.5, 0.55, 'Signal to Noise Limit (dB): ' + "{:.2f}".format(round(20*np.log10(StoNlim),2)) + " +/- " + \
        "{:.3f}".format(round(((20*svn_err)/(StoNlim*np.log(10))),2)), ha='center', va='center', fontsize=8)
    
    ax_text.text(0.5, 0.45, 'Dynamic Range (dB): ' + "{:.2f}".format(round(20 * np.log10(fullWell / readNoise), 2)) +" +/- " + \
        "{:.3f}".format(round((20 / np.log(10)) * np.sqrt((fullWell_err / fullWell)**2 + (readNoise_err / readNoise)**2), 2)), \
            ha='center', va='center', fontsize=8)
    
    ax_text.text(0.5, 0.35, 'Full Well (e-): ' + "{:.2f}".format(round((fullWell * sensitivity), 2)) + " +/- " + "{:.3f}".format(round((fullWell*sensitivity* \
        np.sqrt((fullWell_err/fullWell)**2+ (sensErr/sensitivity)**2)),2)), ha='center', va='center', fontsize=8)
    
    ax_text.text(0.5, 0.25, "Sensitivity (e-/DN): " + "{:.2f}".format(round(sensitivity, 2)) + " +/- " + "{:.3f}".format(round(sensErr,2)), \
        ha='center', va='center', fontsize=8)
    
    ax_text.text(0.5, 0.15, "Pn: " + "{:.2f}".format(round(Pn*100,2)) + " +/- "  + "{:.3f}".format(round(Pn_err*100,2)) + "%", ha='center', va='center', fontsize=8)

    ax_text.text(0.5, 0.05, 'Read Noise (e-): ' + "{:.2f}".format(round(readNoise * sensitivity,2)) + " +/- " + "{:.3f}".format(round((readNoise*sensitivity)* \
        np.sqrt(((readNoise_err/readNoise)**2 + (sensErr/sensitivity)**2)), 3)), ha='center', va='center', fontsize=8)

    fig.canvas.draw()
    # Show the combined figure
    plt.tight_layout()
    fig.subplots_adjust(wspace=0.3, hspace=0.3, bottom=0.15)
    plt.show()
#function call to run program, insert correct folder paths
#generator(<path_to_offsets>, <path_to_illuminated_images>, <path_to_flats>, (rowStart, rowEnd), (colStart, colEnd))