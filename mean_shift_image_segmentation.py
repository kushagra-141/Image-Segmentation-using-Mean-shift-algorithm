# Required libraries
# For numerical operations
import numpy as np  
# For clustering operations
from sklearn.cluster import MeanShift, estimate_bandwidth  
# For image processing
from PIL import Image  
# For OS-level operations like reading file names
import os  

# Define the directory containing the images
image_directory = 'dataset'

# List of all JPG image file paths in the specified directory
all_image_files = [os.path.join(image_directory, fname) for fname in os.listdir(image_directory) if fname.endswith('.jpg')]

# Iterate over each image file in the directory
for image_path in all_image_files:

    # Load the image using PIL and convert it to a numpy array
    img = Image.open(image_path)
    image = np.array(img)

    # Check if the image has three channels (RGB). If not, skip to the next image.
    if image.ndim != 3 or image.shape[2] != 3:
        print(f"Skipping image {image_path} due to incompatible shape: {image.shape}")
        continue

    # Get the shape (dimensions) of the image
    shape = image.shape
    
    # Reshape the image into a 2D array, where each row is a pixel and columns represent RGB values
    reshape_img = np.reshape(image, [-1, 3])

    # Estimate the bandwidth for the MeanShift algorithm. This is a parameter that influences how the clustering is done.
    bandwidth = estimate_bandwidth(reshape_img, quantile=0.1, n_samples=100)

    # Create a MeanShift clustering model with the estimated bandwidth
    msc = MeanShift(bandwidth=bandwidth, bin_seeding=True)

    # Fit the model to the reshaped image data
    msc.fit(reshape_img)

    # Print insights about the clustering for the current image
    print("Processing Image:", image_path)
    print("Number of estimated clusters:", len(np.unique(msc.labels_)),"\n")

    # Assign a label (cluster) to each pixel in the image
    labels = msc.labels_

    # Reshape the labels to match the original image dimensions
    result_image = np.reshape(labels, shape[:2])

    # Save the segmented image after scaling the labels to range between 0 and 255
    segmented_image_path = os.path.join('segmented', os.path.basename(image_path))
    Image.fromarray((result_image * (255 / (len(np.unique(msc.labels_)) - 1))).astype('uint8')).save(segmented_image_path)
