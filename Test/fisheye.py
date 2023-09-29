import cv2
import numpy as np 

def ApplyFishEye(image, focal_length=280, y_up=300, dit = 0.5, y_t = 0.2, mask_np= None, mask_image=None, is_target = False):
    
    height, width = image.shape[:2]
    
    center_x = width / 2 
    center_y = height / 2
    
    camera_matrix = np.array([[focal_length, 0, center_x],
                              [0, focal_length, center_y + y_up ],
                              [0, 0, 1]], dtype=np.float32)
    
    dist_coeffs = np.array([0, dit, 0, 0], dtype=np.float32)
    
    translation_matrix = np.array([[1 -0.05, 0, 0],
                                   [0, 1 - y_t , 0],
                                   [0, 0, 1]], dtype=np.float32)
    
    translated_camera_matrix = np.dot(translation_matrix, camera_matrix)
    # 왜곡 보정
    undistorted_image = cv2.undistort(image, translated_camera_matrix, dist_coeffs)
    
    lower_tone = (1, 1, 1)  # Example lower tone threshold (BGR format)
    upper_tone = (255, 255, 255)  # Example upper tone threshold (BGR format)
    mask = cv2.inRange(undistorted_image, lower_tone, upper_tone)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize variables to track the largest rectangle
    largest_area = 0
    largest_rectangle = None

    # Iterate through the contours and find the largest rectangle
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area > largest_area:
            largest_area = area
            largest_rectangle = (x, y, w, h)
    
    undistorted_image = undistorted_image[y + int(h*0.1) : y + h - int(h*0.1), x + int(w*0.1) : x + w - int(w*0.1)]
    undistorted_image = cv2.resize(undistorted_image, (1024,512))
    if not is_target:
        undistorted_image[~mask_np] = 0
        undistorted_image += mask_image.astype(np.uint8)
    else:
        undistorted_image[~mask_np] = 12
        undistorted_image = cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2GRAY)
    
    return undistorted_image