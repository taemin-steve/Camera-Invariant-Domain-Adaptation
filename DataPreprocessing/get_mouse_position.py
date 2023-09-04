import cv2

# Create a mouse callback function to get and display pixel coordinates
def get_coordinates(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x,y)

# Load the image
image_path = "./DataPreprocessing/edge_detected_image.py"
image = cv2.imread(image_path)

# Create a window and set the mouse callback
cv2.namedWindow("Image")
cv2.setMouseCallback("Image", get_coordinates)

# Display the image
cv2.imshow("Image", image)

# Wait for a key press and close the window
cv2.waitKey(0)
cv2.destroyAllWindows()
