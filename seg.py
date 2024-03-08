import cv2
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
from sklearn.metrics import mean_squared_error

# Load your image
image = cv2.imread(r"C:\Users\HERMEHAR BEDI\OneDrive\Desktop\Hermehar\Programming\Python\Brain\Watershed-kmeans-comparison\brainTumor.png")
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

    # Load the saved Watershed segmented image
    segmented_image = cv2.imread(r"C:\Users\HERMEHAR BEDI\OneDrive\Desktop\Hermehar\Programming\Python\Brain\Watershed-kmeans-comparison\Brain-tumour-segmented.png")
    # Resize the segmented_image and watershed_image to the same dimensions as the original image
    segmented_image = cv2.resize(segmented_image, (image.shape[1], image.shape[0]))
    watershed_image = cv2.resize(kmeans_segmented_image, (image.shape[1], image.shape[0]))

    # Calculate SSIM (Structural Similarity Index) between K-Means Segmentation and Watershed Segmentation
    ssim = compare_ssim(segmented_image, watershed_image, multichannel=True)

    # Display the segmented images
    cv2.imshow('Ground Truth', segmented_image)
    cv2.imshow('Watershed Segmentation', watershed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f"SSIM between K-Means Segmentation and Ground Truth Image: {ssim}")
