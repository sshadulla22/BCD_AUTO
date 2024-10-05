import cv2
import numpy as np
from skimage.filters import threshold_multiotsu
import streamlit as st  # Ensure Streamlit is imported


def preprocess_image(image):
    if len(image.shape) == 3:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image_gray = image.copy()
    
    image_normalized = cv2.normalize(image_gray, None, 0, 255, cv2.NORM_MINMAX)
    return image_normalized

def find_connected_components(image):
    _, binary_image = cv2.threshold(image, 20, 255, cv2.THRESH_BINARY)
    num_labels, labels = cv2.connectedComponents(binary_image)
    return num_labels, labels

def remove_non_breast_areas(image, labels, num_labels):
    mask = np.zeros(image.shape, dtype=np.uint8)
    largest_area = 0
    largest_label = 0

    for label in range(1, num_labels):
        component_mask = (labels == label).astype(np.uint8) * 255
        component_area = cv2.countNonZero(component_mask)

        if component_area > largest_area:
            largest_area = component_area
            largest_label = label

    mask[labels == largest_label] = 255
    return mask

def enhance_image(image):
    enhanced_image = cv2.equalizeHist(image)
    return enhanced_image

def apply_multi_otsu(image):
    thresholds = threshold_multiotsu(image, classes=4)
    regions = np.digitize(image, bins=thresholds)
    return regions

def select_seed_pixel(image):
    height, width = image.shape
    mid_height = height // 2
    mid_width = width // 2
    q1 = image[0:mid_height, 0:mid_width]
    q1_mid_height = mid_height // 2
    q1_mid_width = mid_width // 2
    q11 = q1[0:q1_mid_height, 0:q1_mid_width]
    q11_mid_y = q1_mid_height // 2
    q11_mid_x = q1_mid_width // 2
    seed_pixel = (q11_mid_y, q11_mid_x)
    seed_value = q11[q11_mid_y, q11_mid_x]
    return seed_pixel, seed_value

def dfs_iterative(image, start, seed_value, visited, mask):
    stack = [start]
    
    while stack:
        x, y = stack.pop()
        
        # Check bounds and seed value
        if (x < 0 or x >= len(image) or 
            y < 0 or y >= len(image[0]) or 
            visited[x][y] or 
            image[x][y] != seed_value):  # Only continue if pixel value is the same as seed value
            continue
        
        # Mark the pixel as visited
        visited[x][y] = True
        mask[x][y] = True  # Mark the pixel in the mask

        # Add neighboring pixels (up, down, left, right)
        stack.append((x + 1, y))  # Down
        stack.append((x - 1, y))  # Up
        stack.append((x, y + 1))  # Right
        stack.append((x, y - 1))  # Left

def remove_pectoral_muscle(image, segmented_image, seed_pixel):
    # Create a mask for the connected component
    mask_pectoral = np.zeros(image.shape, dtype=np.uint8)  # Mask for pectoral muscle
    visited = np.zeros(image.shape, dtype=bool)

    # Get the seed value at the seed pixel
    seed_value = segmented_image[seed_pixel]

    # Perform DFS to find the connected component for the pectoral muscle
    dfs_iterative(segmented_image, seed_pixel, seed_value, visited, mask_pectoral)

    # Cut the pectoral muscle from the segmented image to get the breast area
    mask_breast = ~mask_pectoral.astype(bool)  # Invert the mask to get breast area
    image_with_cut_breast = cv2.bitwise_and(image, image, mask=mask_breast.astype(np.uint8))

    # Cut the pectoral muscle from the image
    image_with_cut_pectoral = cv2.bitwise_and(image, image, mask=mask_pectoral)

    return image_with_cut_pectoral, image_with_cut_breast

def extract_dense_regions(segmented_image):
    # Create a mask for the dense regions based on a threshold
    threshold_value = 2  # This can be adjusted based on your requirements
    dense_region_mask = segmented_image >= threshold_value  # Boolean mask where dense regions are located

    # Create an output image showing the dense regions in white on a black background
    dense_region_image = np.zeros_like(segmented_image, dtype=np.uint8)
    dense_region_image[dense_region_mask] = 255  # Set dense regions to white

    return dense_region_image, dense_region_mask

def draw_dense_region_contours(dense_region_image, dense_region_mask):
    contours, _ = cv2.findContours(dense_region_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a blank image to draw contours
    contour_image = np.zeros_like(dense_region_image, dtype=np.uint8)
    
    # Draw contours in blue
    cv2.drawContours(contour_image, contours, -1, (255, 0, 0), 1)  # 1 pixel thick lines

    # Draw contours for high-intensity pixels
    high_intensity_contours, _ = cv2.findContours(dense_region_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(contour_image, high_intensity_contours, -1, (0, 255, 0), 1)  # Green for high intensity
    
    return contour_image

# Streamlit app
def main():
    st.title("Breast Image Processing")
    
    # Upload image
    image_file = st.file_uploader("Upload an image...", type=["png", "jpg", "jpeg"])
    
    if image_file is not None:
        original_image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(original_image, (300, 400))
        
        preprocessed_image = preprocess_image(image)
        num_labels, labels = find_connected_components(preprocessed_image)
        breast_area_mask = remove_non_breast_areas(preprocessed_image, labels, num_labels)
        image_without_artifacts = cv2.bitwise_and(image, breast_area_mask)
        enhanced_image = enhance_image(preprocessed_image)
        segmented_image = apply_multi_otsu(image_without_artifacts)
        seed_pixel, seed_value = select_seed_pixel(segmented_image)

        # Perform DFS and cut the pectoral muscle
        image_pect, image_breast = remove_pectoral_muscle(image_without_artifacts, segmented_image, seed_pixel)

        # Extract dense region image
        dense_region_image, dense_region_mask = extract_dense_regions(segmented_image)
        
        # Draw contours for dense regions and high-intensity pixels
        dense_region_contours = draw_dense_region_contours(dense_region_image, dense_region_mask)

        # Output the result
        st.write(f"Seed Pixel Coordinates: {seed_pixel}")
        st.write(f"Seed Pixel Value: {seed_value}")

        # Display images
        st.subheader('Original Image')
        st.image(original_image, channels="GRAY")

        st.subheader('Enhanced Image')
        st.image(enhanced_image, channels="GRAY")

        st.subheader('Pectoral Muscle Cut')
        st.image(image_pect, channels="GRAY")
        
        st.subheader('Breast Area Cut')
        st.image(image_breast, channels="GRAY")
        
        # Display the dense region contours image
        st.subheader("Dense Region with Contours")
        st.image(dense_region_contours, caption="Dense Regions with Blue Contours and High-Intensity Pixels in Green", use_column_width=False, width=400)

if __name__ == "__main__":
    main()
