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




# Function to create and plot any random polygon
def plot_random_polygon(polygon_points, roll1, pitch1, yaw1, roll2, pitch2, yaw2, scale=1.0, center=(400, 400)):
    image = np.zeros((800, 800), dtype=np.uint8)
    
    # Apply first rotation
    rotation_matrix1 = get_rotation_matrix(roll1, pitch1, yaw1)
    rotated_coords = rotate_coords(polygon_points - center, rotation_matrix1)  # Centered rotation
    
    # Apply second rotation
    rotation_matrix2 = get_rotation_matrix(roll2, pitch2, yaw2)
    rotated_coords = rotate_coords(rotated_coords, rotation_matrix2)  # Centered rotation
    
    # Apply scaling and translate back to center
    rotated_coords = (rotated_coords - rotated_coords.min(axis=0)) * scale + center
    
    rotated_coords = rotated_coords.astype(np.int32)
    
    # # Fill the polygon in the image
    # cv2.fillPoly(image, [rotated_coords], 255)


    for point in rotated_coords:
        # Draw a circle at each vertex
        cv2.circle(image, (int(point[0]), int(point[1])), radius=3, color=255, thickness=-1)


    return image


def load_polygon_points(txt_path, image_size=(800, 800), downsample_factor=30):
    """
    Load, rescale, and optionally downsample polygon points from a text file to fit within the image size.

    Args:
        txt_path (str): Path to the text file containing the polygon points.
        image_size (tuple): Size of the target image (width, height).
        downsample_factor (int): Factor to downsample the points. Keep every nth point.

    Returns:
        np.ndarray: Array of rescaled and downsampled polygon points.
    """
    try:
        points = []
        with open(txt_path, "r") as file:
            for line in file:
                # Skip lines that are headers or do not contain numeric data
                line = line.strip()
                if not line or ":" in line or "x" in line or "y" in line:
                    continue
                
                # Parse numeric data
                coords = line.split()
                if len(coords) == 2:  # Ensure there are two values
                    x, y = map(float, coords)
                    points.append([x, y])
        
        if not points:
            raise ValueError("No valid points found in the file.")
        
        points = np.array(points, dtype=np.float32)

        # Downsample the points
        downsampled_points = points[::downsample_factor]

        # Rescale points to fit within the image size
        min_coords = downsampled_points.min(axis=0)
        max_coords = downsampled_points.max(axis=0)
        range_coords = max_coords - min_coords

        scale = min(image_size[0] / range_coords[0], image_size[1] / range_coords[1])
        rescaled_points = (downsampled_points - min_coords) * scale

        return rescaled_points
    except Exception as e:
        print(f"Error loading points: {e}")
        return None






def adjust_polygon_edges(polygon_points, top_adjust, bottom_adjust):
    """
    Adjust the polygon by squeezing or expanding horizontally (left-right) for the top and bottom portions of the polygon.
    
    Args:
        polygon_points (np.ndarray): Original polygon points.
        top_adjust (float): Adjustment factor for the top side of the polygon. 
                            Positive values will expand the top; negative values will squeeze the top.
        bottom_adjust (float): Adjustment factor for the bottom side of the polygon. 
                               Positive values will expand the bottom; negative values will squeeze the bottom.
    
    Returns:
        np.ndarray: Adjusted polygon points.
    """
    # Find the min and max y-coordinates of the polygon
    y_min = polygon_points[:, 1].min()
    y_max = polygon_points[:, 1].max()

    # Calculate the total height of the polygon
    total_height = y_max - y_min

    # Find the midpoint of the polygon in the y-direction
    y_center = (y_min + y_max) / 2

    # Apply the top adjustment to the polygon
    adjusted_polygon = polygon_points.copy()

    # Adjust the top half
    for i, point in enumerate(polygon_points):
        y = point[1]
        x = point[0]

        # Only adjust points in the top half of the polygon (above the center)
        if y < y_center:  # Top part of the polygon
            # Calculate the horizontal distance from the center
            x_center = np.mean(polygon_points[polygon_points[:, 1] < y_center][:, 0])
            distance_from_center = x - x_center

            # Apply the horizontal adjustment for the top part
            adjusted_polygon[i, 0] -= top_adjust * (distance_from_center) * (1 - (y - y_min) / total_height)

    # Adjust the bottom half
    for i, point in enumerate(polygon_points):
        y = point[1]
        x = point[0]

        # Only adjust points in the bottom half of the polygon (below the center)
        if y > y_center:  # Bottom part of the polygon
            # Calculate the horizontal distance from the center
            x_center = np.mean(polygon_points[polygon_points[:, 1] > y_center][:, 0])
            distance_from_center = x - x_center

            # Apply the horizontal adjustment for the bottom part
            adjusted_polygon[i, 0] += bottom_adjust * (distance_from_center) * (1 - (y_max - y) / total_height)

    return adjusted_polygon



# Path to the text file containing polygon points
polygon_file = "sample_pts.txt"  # Replace with your file path
polygon_points = load_polygon_points(polygon_file)


def rotate_polygon(polygon_points):
    """
    Rotates the polygon points by 90 degrees to the right (clockwise).

    Args:
        polygon_points (np.ndarray): Original polygon points.
    
    Returns:
        np.ndarray: Rotated polygon points.
    """
    # Rotation matrix for 90 degrees clockwise
    rotation_matrix = np.array([[0, 1],
                                [-1, 0]])

    # Apply the rotation to each point
    rotated_polygon = np.dot(polygon_points, rotation_matrix.T)
    
    return rotated_polygon


polygon_points = rotate_polygon(polygon_points)


if polygon_points is None:
    print("No points found. Exiting.")
    exit()


# # Manually define the square points
# polygon_points = np.array([
#     [380, 300],  # 1
#     [420, 300],  # 2
#     [380, 350],  # 3
#     [420, 350],  # 4
#     [380, 450],  # 5
#     [420, 450],  # 6
#     [380, 500],  # 7
#     [420, 500],  # 8
#     [300, 380],  # 9
#     [350, 380],  # 10
#     [450, 380],  # 11
#     [500, 380],  # 12
#     [300, 420],  # 13
#     [350, 420],  # 14
#     [450, 420],  # 15
#     [500, 420]   # 16
# ])



# Create a window for the image with fixed size
cv2.namedWindow('Random Polygon', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Random Polygon', 800, 800)  # Set a fixed window size

# Create trackbars for roll, pitch, yaw (for both rotations)
cv2.createTrackbar("Roll1 (Origin)", "Random Polygon", 0, 360, lambda x: None)
cv2.createTrackbar("Pitch1 (Origin)", "Random Polygon", 0, 360, lambda x: None)
cv2.createTrackbar("Yaw1 (Origin)", "Random Polygon", 0, 360, lambda x: None)

cv2.createTrackbar("Roll2 (Center)", "Random Polygon", 0, 360, lambda x: None)
cv2.createTrackbar("Pitch2 (Center)", "Random Polygon", 0, 360, lambda x: None)
cv2.createTrackbar("Yaw2 (Center)", "Random Polygon", 0, 360, lambda x: None)

# Create trackbars for moving the polygon (left-right and up-down)
cv2.createTrackbar("X Position", "Random Polygon", 400, 800, lambda x: None)  # Horizontal position
cv2.createTrackbar("Y Position", "Random Polygon", 400, 800, lambda x: None)  # Vertical position

# Create trackbars for squeezing or expanding the top and bottom edges
cv2.createTrackbar("Top Squeeze/Expand", "Random Polygon", 400, 800, lambda x: None)
cv2.createTrackbar("Bottom Squeeze/Expand", "Random Polygon", 400, 800, lambda x: None)



while True:
    # Get the current positions of the trackbars for both rotations
    roll1 = cv2.getTrackbarPos("Roll1 (Origin)", "Random Polygon")
    pitch1 = cv2.getTrackbarPos("Pitch1 (Origin)", "Random Polygon")
    yaw1 = cv2.getTrackbarPos("Yaw1 (Origin)", "Random Polygon")
    
    roll2 = cv2.getTrackbarPos("Roll2 (Center)", "Random Polygon")
    pitch2 = cv2.getTrackbarPos("Pitch2 (Center)", "Random Polygon")
    yaw2 = cv2.getTrackbarPos("Yaw2 (Center)", "Random Polygon")
    
    # Get the current position of the center (from X and Y position trackbars)
    x_position = cv2.getTrackbarPos("X Position", "Random Polygon")
    y_position = cv2.getTrackbarPos("Y Position", "Random Polygon")
    
    # Get the top and bottom adjustment factors
    top_adjust = cv2.getTrackbarPos("Top Squeeze/Expand", "Random Polygon") - 400
    bottom_adjust = cv2.getTrackbarPos("Bottom Squeeze/Expand", "Random Polygon") - 400
    
    # Update the center position based on trackbar values
    center = (x_position, y_position)
    
    # Adjust the top and bottom edges of the polygon
    adjusted_polygon_points = adjust_polygon_edges(polygon_points, top_adjust, bottom_adjust)

    # Create and plot the random polygon with the current roll, pitch, yaw values for both rotations
    image = plot_random_polygon(adjusted_polygon_points, roll1, pitch1, yaw1, roll2, pitch2, yaw2, center=center)
    
    # Display the image
    cv2.imshow("Random Polygon", image)
    
    # Exit if the user presses the 'Esc' key
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
