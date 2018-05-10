import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image = mpimg.imread('./cutouts/8.png')


# Define a function to compute color histogram features
def color_hist(img, nbins=32, bins_range=(0, 256)):
    img_copy = np.copy(img)
    # Compute the histogram of the RGB channels separately
    rhist = np.histogram(img_copy[:,:, 0], bins=32, range=(0, 256))
    ghist = np.histogram(img_copy[:,:, 1], bins=32, range=(0, 256))
    bhist = np.histogram(img_copy[:,:, 2], bins=32, range=(0, 256))
    # Generating bin centers
    bin_start = rhist[1]
    bin_centers = (bin_start[1:] + bin_start[:len(bin_start)-1])/2
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return rhist, ghist, bhist, bin_centers, hist_features


rh, gh, bh, bincen, feature_vec = color_hist(image, nbins=32, bins_range=(0, 256))

# Plot a figure with all three bar charts
if rh is not None:
    fig = plt.figure(figsize=(12, 3))
    plt.subplot(131)
    plt.bar(bincen, rh[0])
    plt.xlim(0, 256)
    plt.title('R Histogram')
    plt.subplot(132)
    plt.bar(bincen, gh[0])
    plt.xlim(0, 256)
    plt.title('G Histogram')
    plt.subplot(133)
    plt.bar(bincen, bh[0])
    plt.xlim(0, 256)
    plt.title('B Histogram')
    fig.tight_layout()
    plt.show()
else:
    print('Your function is returning None for at least one variable...')