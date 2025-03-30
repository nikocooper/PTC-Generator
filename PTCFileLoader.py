import glob
import os
import numpy as np
from astropy.io import fits

#extracts 2d fits files from a folder
def extractFits(folderPath):
    fitsFiles = glob.glob(os.path.join(folderPath, "*.fits"))
    imageList = np.array([fits.getdata(file) for file in fitsFiles])
    print(f"Loaded {len(imageList)} FITS files")
    return imageList

#extracts 3d fits files from a folder
def extractFitsCubes(folderPath):
    # Find all FITS files in the folder
    fitsFiles = glob.glob(os.path.join(folderPath, "*.fits"))
    
    # Load all FITS cubes into a list
    cubes = [fits.getdata(file) for file in fitsFiles]

    # Ensure all FITS cubes have the same shape
    if not all(cube.shape == cubes[0].shape for cube in cubes):
        raise ValueError("All FITS cubes must have the same dimensions.")

    # Convert list of cubes to a NumPy array (Shape: [num_cubes, depth, height, width])
    stacked_images = np.array(cubes)

    print(f"Loaded {len(fitsFiles)} FITS cubes with shape {stacked_images.shape}")
    
    return stacked_images

# returns the mean image of each cube in a folder
def extractMeanFromCubes(folderPath):
    fitsFiles = glob.glob(os.path.join(folderPath, "*.fits"))
    imageList = []
    for file in fitsFiles:
        with fits.open(file) as hdul:
            data = hdul[0].data
            if data.ndim == 3:  # Check if the data is a 3D array (cube)
                mean_image = np.mean(data, axis=0)  # Take the mean across the first axis
                imageList.append(mean_image)
    return np.array(imageList)