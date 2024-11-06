import cv2
import numpy as np

# Function to process image and determine traffic light color
def determine_traffic_light_color(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Threshold to get the light mask
    _, light_mask = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
    
    # Connected components to find the light boxes
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(light_mask)
    
    # Filter out small components
    min_size = 1000
    large_labels = [i for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] >= min_size]
    
    # Create an output image to visualize the different connected areas
    output_image = np.zeros_like(image)
    height, width = light_mask.shape
    
    roundness_scores = []
    
    for i in large_labels:
        x, y, w, h, area = stats[i]
        if y + h < height and x + w < width and x > 0 and y > 0:  # Ensure the mask does not touch the image borders
            mask = (labels == i).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cnt = contours[0]
                perimeter = cv2.arcLength(cnt, True)
                roundness = (4 * np.pi * area) / (perimeter ** 2)
                roundness_scores.append((roundness, i))
    
    # Sort by roundness score
    roundness_scores.sort(reverse=True)
    
    if roundness_scores:
        if roundness_scores[0][0] > 0.6:
            best_label = roundness_scores[0][1]
            best_mask = (labels == best_label).astype(np.uint8) * 255
            
            # Find the centroid of the best mask
            M = cv2.moments(best_mask)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0
            
            # Calculate the radius of the light mask
            radius = int(np.sqrt(np.sum(best_mask) / (np.pi*255)))
            
            # Create a rectangular mask centered on the light mask
            rect_mask = np.zeros_like(light_mask)
            rect_width = radius * 6
            rect_height = radius * 2
            top_left = (max(cX - rect_width // 2, 0), max(cY - rect_height // 2, 0))
            bottom_right = (min(cX + rect_width // 2, width), min(cY + rect_height // 2, height))
            cv2.rectangle(rect_mask, top_left, bottom_right, 255, -1)
            
            # Invert the masks
            inv_black_mask = cv2.bitwise_not(light_mask)
            inv_rect_mask = cv2.bitwise_not(rect_mask)
            
            # Find the overlap between the inverted masks
            overlap = cv2.bitwise_and(inv_black_mask, inv_rect_mask)
            
            # Determine the position of the overlap within the rectangular mask
            overlap_coords = np.column_stack(np.where(overlap == 255))
            
            if len(overlap_coords) > 0:
                avg_overlap_x = np.mean(overlap_coords[:, 0])
                print(avg_overlap_x, top_left[0], bottom_right[0])
                if avg_overlap_x < top_left[0] + rect_width * 1/3:
                    color_name = "Green light"
                elif avg_overlap_x > bottom_right[0] - rect_width * 1/3:
                    color_name = "Red light"
                else:
                    color_name = "Amber light"
            else:
                color_name = "Bright spot not found within traffic light box"
            
            # Display the original light mask and the output image with colored masks
            cv2.imshow('Light Mask', light_mask)
            cv2.imshow('Rectangular Mask', rect_mask)
            cv2.imshow('Overlap', overlap)
            cv2.waitKey(5000)
            cv2.destroyAllWindows()
            
            return color_name
    else:
        return "Bright spot not found within traffic light box"

# Load images
image1_path = "/home/innodriver/InnoDriver_ws/src/missionRacing/src/1721683282400511503.jpg"
image2_path = "/home/innodriver/InnoDriver_ws/src/missionRacing/src/1721683286900440692.jpg"
image3_path = "/home/innodriver/InnoDriver_ws/src/missionRacing/src/1721683290900520801.jpg"

images = [cv2.imread(image1_path), cv2.imread(image2_path), cv2.imread(image3_path)]

# Process each image and determine the traffic light color
results = [determine_traffic_light_color(image) for image in images]

print(results)
