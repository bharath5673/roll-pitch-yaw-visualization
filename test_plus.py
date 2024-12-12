import cv2
import numpy as np

# Function to create the rotation matrices for roll, pitch, and yaw
def get_rotation_matrix(roll, pitch, yaw):
    # Convert angles to radians
    roll_rad = np.radians(roll)
    pitch_rad = np.radians(pitch)
    yaw_rad = np.radians(yaw)
    
    # Roll rotation matrix (rotation around x-axis)
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(roll_rad), -np.sin(roll_rad)],
        [0, np.sin(roll_rad), np.cos(roll_rad)]
    ])
    
    # Pitch rotation matrix (rotation around y-axis)
    R_y = np.array([
        [np.cos(pitch_rad), 0, np.sin(pitch_rad)],
        [0, 1, 0],
        [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]
    ])
    
    # Yaw rotation matrix (rotation around z-axis)
    R_z = np.array([
        [np.cos(yaw_rad), -np.sin(yaw_rad), 0],
        [np.sin(yaw_rad), np.cos(yaw_rad), 0],
        [0, 0, 1]
    ])
    
    # Combined rotation matrix
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R


# Function to rotate the coordinates based on the roll, pitch, yaw
def rotate_coords(coords, rotation_matrix, center=None):
    rotated_coords = []
    for point in coords:
        if center:
            # Translate to origin, apply rotation, then translate back
            translated_point = np.array([point[0] - center[0], point[1] - center[1], 0])
            rotated_point = np.dot(rotation_matrix, translated_point)
            rotated_point += np.array([center[0], center[1], 0])  # Translate back
        else:
            # Rotation around origin (0, 0)
            rotated_point = np.dot(rotation_matrix, np.array([point[0], point[1], 0]))
        
        rotated_coords.append([rotated_point[0], rotated_point[1]])
    
    return np.array(rotated_coords, dtype=np.int32)


# Function to create the plus sign and apply rotation
def create_plus_sign(roll1, pitch1, yaw1, roll2, pitch2, yaw2, scale=1.0, center=(400, 400)):
    image = np.zeros((800, 800), dtype=np.uint8)
    thickness = 30
    size = 250
    
    vertical_coords = np.array([
        [center[0] - thickness // 2, center[1] - size // 2],
        [center[0] + thickness // 2, center[1] - size // 2],
        [center[0] + thickness // 2, center[1] + size // 2],
        [center[0] - thickness // 2, center[1] + size // 2]
    ])
    
    horizontal_coords = np.array([
        [center[0] - size // 2, center[1] - thickness // 2],
        [center[0] + size // 2, center[1] - thickness // 2],
        [center[0] + size // 2, center[1] + thickness // 2],
        [center[0] - size // 2, center[1] + thickness // 2]
    ])
    
    rotation_matrix1 = get_rotation_matrix(roll1, pitch1, yaw1)
    vertical_coords_rotated = rotate_coords(vertical_coords, rotation_matrix1)
    horizontal_coords_rotated = rotate_coords(horizontal_coords, rotation_matrix1)
    
    rotation_matrix2 = get_rotation_matrix(roll2, pitch2, yaw2)
    vertical_coords_rotated = rotate_coords(vertical_coords_rotated, rotation_matrix2, center)
    horizontal_coords_rotated = rotate_coords(horizontal_coords_rotated, rotation_matrix2, center)
    
    vertical_coords_rotated = (vertical_coords_rotated - center) * scale + center
    horizontal_coords_rotated = (horizontal_coords_rotated - center) * scale + center
    
    vertical_coords_rotated = vertical_coords_rotated.astype(np.int32)
    horizontal_coords_rotated = horizontal_coords_rotated.astype(np.int32)
    
    cv2.fillPoly(image, [vertical_coords_rotated], 255)
    cv2.fillPoly(image, [horizontal_coords_rotated], 255)

    return image

# Create a window for the image with fixed size
cv2.namedWindow('Plus Sign', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Plus Sign', 800, 800)  # Set a fixed window size

# Create trackbars for roll, pitch, yaw (for both rotations)
cv2.createTrackbar("Roll1 (Origin)", "Plus Sign", 0, 360, lambda x: None)
cv2.createTrackbar("Pitch1 (Origin)", "Plus Sign", 0, 360, lambda x: None)
cv2.createTrackbar("Yaw1 (Origin)", "Plus Sign", 0, 360, lambda x: None)

cv2.createTrackbar("Roll2 (Center)", "Plus Sign", 0, 360, lambda x: None)
cv2.createTrackbar("Pitch2 (Center)", "Plus Sign", 0, 360, lambda x: None)
cv2.createTrackbar("Yaw2 (Center)", "Plus Sign", 0, 360, lambda x: None)

# Create trackbars for moving the plus sign (left-right and up-down)
cv2.createTrackbar("X Position", "Plus Sign", 400, 800, lambda x: None)  # Horizontal position
cv2.createTrackbar("Y Position", "Plus Sign", 400, 800, lambda x: None)  # Vertical position

while True:
    # Get the current positions of the trackbars for both rotations
    roll1 = cv2.getTrackbarPos("Roll1 (Origin)", "Plus Sign")
    pitch1 = cv2.getTrackbarPos("Pitch1 (Origin)", "Plus Sign")
    yaw1 = cv2.getTrackbarPos("Yaw1 (Origin)", "Plus Sign")
    
    roll2 = cv2.getTrackbarPos("Roll2 (Center)", "Plus Sign")
    pitch2 = cv2.getTrackbarPos("Pitch2 (Center)", "Plus Sign")
    yaw2 = cv2.getTrackbarPos("Yaw2 (Center)", "Plus Sign")
    
    # Get the current position of the center (from X and Y position trackbars)
    x_position = cv2.getTrackbarPos("X Position", "Plus Sign")
    y_position = cv2.getTrackbarPos("Y Position", "Plus Sign")
    
    # Create the plus sign with the current roll, pitch, yaw values for both rotations
    image = create_plus_sign(roll1, pitch1, yaw1, roll2, pitch2, yaw2, center=(x_position, y_position))
    
    # Display the image
    cv2.imshow("Plus Sign", image)
    
    # Exit if the user presses the 'Esc' key
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
