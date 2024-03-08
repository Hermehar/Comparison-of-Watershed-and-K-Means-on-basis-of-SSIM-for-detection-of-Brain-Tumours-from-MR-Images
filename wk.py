import cv2
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
from sklearn.metrics import mean_squared_error

# Load your image
image = cv2.imread(r"C:\Users\HERMEHAR BEDI\OneDrive\Desktop\Hermehar\Programming\Python\Brain\mri2.jpg")
# Check if the image is loaded successfully
if image is None:
    print("Error: Image not loaded.")
else:
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # K-Means Segmentation
    # Reshape the image into a 2D array of pixels
    pixels = image.reshape((-1, 3))
    pixels = np.float32(pixels)

    # Define K-Means criteria and apply K-Means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    k = 3  # The number of clusters (you can adjust this)

    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Convert the centers back to 8-bit values
    centers = np.uint8(centers)

    # Map the labels to their respective centers
    kmeans_segmented = centers[labels.flatten()]
    kmeans_segmented = kmeans_segmented.reshape(image.shape)

    # Save the K-Means segmented image
    cv2.imwrite(r"C:\Users\HERMEHAR BEDI\OneDrive\Desktop\Hermehar\Programming\Python\Brain\Watershed-kmeans-comparison\kmeans_segmented.jpg", kmeans_segmented)

    # Load the saved K-Means segmented image
    kmeans_segmented_image = cv2.imread(r"C:\Users\HERMEHAR BEDI\OneDrive\Desktop\Hermehar\Programming\Python\Brain\Watershed-kmeans-comparison\kmeans_segmented.jpg")

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
    cv2.watershed(image, markers)

    # Mark the segmented regions with a red boundary
    image[markers == -1] = [0, 0, 255]

    # Save the Watershed segmented image
    cv2.imwrite(r"C:\Users\HERMEHAR BEDI\OneDrive\Desktop\Hermehar\Programming\Python\Brain\Watershed-kmeans-comparison\watershed.jpg", image)

    # Load the saved Watershed segmented image
    watershed_image = cv2.imread(r"C:\Users\HERMEHAR BEDI\OneDrive\Desktop\Hermehar\Programming\Python\Brain\Watershed-kmeans-comparison\watershed.jpg")

    # Calculate SSIM (Structural Similarity Index) between K-Means Segmentation and Watershed Segmentation
    ssim = compare_ssim(kmeans_segmented_image, watershed_image, multichannel=True)

    # Display the segmented images
    cv2.imshow('K-Means Segmentation', kmeans_segmented_image)
    cv2.imshow('Watershed Segmentation', watershed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f"SSIM between K-Means Segmentation and Watershed Segmentation: {ssim}")
