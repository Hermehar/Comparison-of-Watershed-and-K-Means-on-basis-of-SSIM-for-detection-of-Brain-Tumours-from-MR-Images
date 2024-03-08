import cv2
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim

# Load your original image
original_image = cv2.imread(r"C:\Users\HERMEHAR BEDI\OneDrive\Desktop\Hermehar\Programming\Python\Brain\Watershed-kmeans-comparison\brainTumor.png")

# Check if the original image is loaded successfully
if original_image is None:
    print("Error: Original Image not loaded.")
else:
    # Convert the original image to grayscale
    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Watershed Segmentation
    # Apply morphological operations to the thresholded image
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=2)

    # Sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # Finding the unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labeling
    _, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0 but 1
    markers = markers + 1

    # Mark the region of the unknown with 0
    markers[unknown == 255] = 0

    # Apply the Watershed algorithm
    cv2.watershed(original_image, markers)

    # Mark the segmented regions with a red boundary
    # original_image[markers == -1] = [0, 0, 255]

    # Save the Watershed segmented image
    cv2.imwrite(r"C:\Users\HERMEHAR BEDI\OneDrive\Desktop\Hermehar\Programming\Python\Brain\Watershed-kmeans-comparison\watershed.png", original_image)

    # Load the saved Watershed segmented image
    watershed_image = cv2.imread(r"C:\Users\HERMEHAR BEDI\OneDrive\Desktop\Hermehar\Programming\Python\Brain\Watershed-kmeans-comparison\watershed.png")
    segmented_image = cv2.imread(r"C:\Users\HERMEHAR BEDI\OneDrive\Desktop\Hermehar\Programming\Python\Brain\Watershed-kmeans-comparison\Brain-tumour-segmented.png")
    # Resize the segmented_image and watershed_image to the same dimensions as the original image
    segmented_image = cv2.resize(segmented_image, (original_image.shape[1], original_image.shape[0]))
    watershed_image = cv2.resize(watershed_image, (original_image.shape[1], original_image.shape[0]))

    # Calculate SSIM (Structural Similarity Index) between K-Means Segmentation and Watershed Segmentation
    ssim = compare_ssim(segmented_image, watershed_image, multichannel=True)

    # Display the segmented images
    cv2.imshow('Ground Truth', segmented_image)
    cv2.imshow('Watershed Segmentation', watershed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f"SSIM between Ground Truth and Watershed Segmentation: {ssim}")