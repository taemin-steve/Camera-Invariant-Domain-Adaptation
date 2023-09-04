import cv2
import numpy as np

# Load your template image
template = cv2.imread('./DataPreprocessing/TRAIN_TARGET_0001.png', 0)
template = template[908:1074, 136:428]

cv2.imshow("1", template)
cv2.waitKey(0)


# Load your collection of images (you would have a list of image filenames here)
image_filenames = ['image1.jpg', 'image2.jpg', 'image3.jpg']

for image_filename in image_filenames:
    # Load the current image from the collection
    image = cv2.imread(image_filename, 0)

    # Perform template matching
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    
    # Set a threshold for matches (adjust as needed)
    threshold = 0.8
    locations = np.where(result >= threshold)

    if locations[0].size > 0:
        print(f"Template found in {image_filename}")
        # You can also access the locations and confidence scores if needed
