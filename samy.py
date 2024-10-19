import cv2
import numpy as np
from skimage.filters import threshold_multiotsu
import streamlit as st
import aiohttp
import asyncio
import base64

#packages for Model
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image


# Placeholder for loading your content moderation model
# from your_moderation_library import load_model
# model = load_model('path_to_your_model')

async def run_model(image_data):
    """
    Run the moderation model on the given image data asynchronously.
    
    Parameters:
        image_data: The image data to be moderated.

    Returns:
        Moderation results from the model.
    """
    # Convert the image data to JPEG and then to base64 format
    _, encoded_image = cv2.imencode('.jpg', image_data)
    image_base64 = base64.b64encode(encoded_image).decode('utf-8')

    # Define the moderation API endpoint and headers
    url = 'https://models.aixplain.com/api/v1/execute/60ddef9c8d38c51c5885e48f'
    headers = {
        'x-api-key': '799b8640ed5d2e45959f34bc3adf4f4c45515d0d492d171b8e7f07cd0da48c1e',
        'content-type': 'application/json'
    }
    body = {
        'image': image_base64,
    }

    # Send a POST request to the moderation API
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=body, headers=headers) as response:
            results = await response.json()  # Get the initial response
            url_to_poll = results['data']     # Extract URL to poll for results
            return await poll_for_results(url_to_poll)  # Poll for results

async def poll_for_results(url_to_poll):
    """
    Poll the moderation API for results until the processing is complete.
    
    Parameters:
        url_to_poll: The URL to poll for the results.

    Returns:
        The completed moderation results.
    """
    async with aiohttp.ClientSession() as session:
        while True:
            async with session.get(url_to_poll, headers={'x-api-key': '799b8640ed5d2e45959f34bc3adf4f4c45515d0d492d171b8e7f07cd0da48c1e'}) as status_response:
                results = await status_response.json()  # Get the results from the API
                if results['completed']:
                    return results['data']  # Return the completed results
            await asyncio.sleep(5)  # Wait before polling again

def preprocess_image(image):
    """
    Preprocess the input image by converting it to grayscale and normalizing it.
    
    Parameters:
        image: The input image.

    Returns:
        Normalized grayscale image.
    """
    # Convert image to grayscale if it is in color
    if len(image.shape) == 3:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image_gray = image.copy()  # Already grayscale
    # Normalize the grayscale image
    image_normalized = cv2.normalize(image_gray, None, 0, 255, cv2.NORM_MINMAX)
    return image_normalized

def find_connected_components(image):
    """
    Find connected components in the binary image.
    
    Parameters:
        image: The input binary image.

    Returns:
        num_labels: The number of connected components.
        labels: An array labeling each connected component.
    """
    _, binary_image = cv2.threshold(image, 20, 255, cv2.THRESH_BINARY)
    num_labels, labels = cv2.connectedComponents(binary_image)  # Identify connected components
    return num_labels, labels

def remove_non_breast_areas(image, labels, num_labels):
    """
    Create a mask for the largest connected component (breast area) in the image.
    
    Parameters:
        image: The input image.
        labels: The labels of connected components.
        num_labels: The number of connected components.

    Returns:
        mask: A binary mask of the breast area.
    """
    mask = np.zeros(image.shape, dtype=np.uint8)  # Initialize mask
    largest_area = 0
    largest_label = 0

    # Loop through each connected component
    for label in range(1, num_labels):
        component_mask = (labels == label).astype(np.uint8) * 255  # Create a mask for the component
        component_area = cv2.countNonZero(component_mask)  # Count non-zero pixels (area)

        # Check if this component is the largest found so far
        if component_area > largest_area:
            largest_area = component_area
            largest_label = label

    mask[labels == largest_label] = 255  # Set the largest component in the mask
    return mask

def enhance_image(image):
    """
    Enhance the contrast of the input image using histogram equalization.
    
    Parameters:
        image: The input image.

    Returns:
        enhanced_image: The contrast-enhanced image.
    """
    if len(image.shape) == 3:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale if needed
    else:
        image_gray = image
    enhanced_image = cv2.equalizeHist(image_gray)  # Apply histogram equalization
    return enhanced_image

def apply_multi_otsu(image):
    """
    Apply Multi-Otsu thresholding to segment the image into regions.
    
    Parameters:
        image: The input image.

    Returns:
        regions: The segmented regions based on Multi-Otsu thresholds.
    """
    thresholds = threshold_multiotsu(image, classes=4)  # Determine thresholds for segmentation
    regions = np.digitize(image, bins=thresholds)  # Segment the image based on thresholds
    return regions

def select_seed_pixel(image):
    """
    Select a seed pixel for region growing based on the center of the image.
    
    Parameters:
        image: The input image.

    Returns:
        seed_pixel: The coordinates of the seed pixel.
        seed_value: The value of the pixel at the seed location.
    """
    height, width = image.shape
    mid_height = height // 2
    mid_width = width // 2
    q1 = image[0:mid_height, 0:mid_width]  # Top-left quadrant
    q1_mid_height = mid_height // 2
    q1_mid_width = mid_width // 2
    q11 = q1[0:q1_mid_height, 0:q1_mid_width]  # Top-left of the top-left quadrant
    q11_mid_y = q1_mid_height // 2
    q11_mid_x = q1_mid_width // 2
    seed_pixel = (q11_mid_y, q11_mid_x)  # Select the seed pixel
    seed_value = q11[q11_mid_y, q11_mid_x]  # Get the pixel value at the seed location
    return seed_pixel, seed_value

def dfs_iterative(image, start, seed_value, visited, mask):
    """
    Perform iterative depth-first search (DFS) to segment the image.
    
    Parameters:
        image: The input image.
        start: The starting coordinates for DFS.
        seed_value: The value of the seed pixel.
        visited: An array tracking visited pixels.
        mask: The mask to mark the segmented region.
    """
    stack = [start]  # Initialize stack with the seed pixel
    
    while stack:
        x, y = stack.pop()  # Pop a pixel from the stack
        
        # Check boundaries and whether the pixel has been visited or matches the seed value
        if (x < 0 or x >= image.shape[0] or 
            y < 0 or y >= image.shape[1] or 
            visited[x, y] or 
            image[x, y] != seed_value):
            continue  # Skip if conditions are not met
        
        visited[x, y] = True  # Mark pixel as visited
        mask[x, y] = True  # Mark pixel in the mask

        # Push adjacent pixels onto the stack
        stack.append((x + 1, y))  # Down
        stack.append((x - 1, y))  # Up
        stack.append((x, y + 1))  # Right
        stack.append((x, y - 1))  # Left

def remove_pectoral_muscle(image, segmented_image, seed_pixel):
    """
    Remove the pectoral muscle region from the image using region growing.
    
    Parameters:
        image: The original image.
        segmented_image: The segmented image after Multi-Otsu.
        seed_pixel: The starting pixel for the DFS.

    Returns:
        image_with_cut_pectoral: Image with pectoral muscle region removed.
        image_with_cut_breast: Image showing the breast area.
    """
    mask_pectoral = np.zeros(image.shape[:2], dtype=np.uint8)  # Initialize mask for pectoral muscle
    visited = np.zeros(image.shape[:2], dtype=bool)  # Track visited pixels

    seed_value = segmented_image[seed_pixel]  # Get the value of the seed pixel

    dfs_iterative(segmented_image, seed_pixel, seed_value, visited, mask_pectoral)  # Perform DFS

    mask_breast = ~mask_pectoral.astype(bool)  # Invert mask to get the breast area
    image_with_cut_breast = cv2.bitwise_and(image, image, mask=mask_breast.astype(np.uint8))  # Extract breast area
    image_with_cut_pectoral = cv2.bitwise_and(image, image, mask=mask_pectoral)  # Extract pectoral muscle area

    return image_with_cut_pectoral, image_with_cut_breast

def extract_dense_regions(segmented_image):
    """
    Extract regions of dense breast tissue from the segmented image.
    
    Parameters:
        segmented_image: The segmented image.

    Returns:
        dense_region_image: Image highlighting dense regions.
        dense_region_mask: Boolean mask of dense regions.
    """
    threshold_value = 2  # Set threshold for dense region detection
    dense_region_mask = segmented_image >= threshold_value  # Create mask for dense regions

    dense_region_image = np.zeros_like(segmented_image, dtype=np.uint8)  # Initialize dense region image
    dense_region_image[dense_region_mask] = 255  # Highlight dense regions in the image

    return dense_region_image, dense_region_mask

def draw_dense_region_contours(dense_region_image, dense_region_mask):
    """
    Draw contours around dense regions in the image.
    
    Parameters:
        dense_region_image: The image highlighting dense regions.
        dense_region_mask: Boolean mask of dense regions.

    Returns:
        contour_image: Image with drawn contours.
    """
    contours, _ = cv2.findContours(dense_region_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find contours
    contour_image = np.zeros_like(dense_region_image, dtype=np.uint8)  # Initialize contour image
    
    # Draw contours of dense regions
    cv2.drawContours(contour_image, contours, -1, (255, 0, 0), 1)  # Draw dense region contours in blue

    high_intensity_contours, _ = cv2.findContours(dense_region_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(contour_image, high_intensity_contours, -1, (0, 255, 0), 1)  # Draw high-intensity contours in green

    return contour_image

def find_highest_dense_region(image):
    _, thresholded_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return thresholded_image, image

    largest_contour = max(contours, key=cv2.contourArea)
    dense_mask = np.zeros_like(image)
    cv2.drawContours(dense_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    
    highest_dense_image = cv2.bitwise_and(image, image, mask=dense_mask)
    dense_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(dense_image, [largest_contour], -1, (0, 0, 255), 2)

    return thresholded_image, dense_image

def moderate_image(image):
    """
    Moderate the image using a content moderation model (mock implementation).
    
    Parameters:
        image: The input image.

    Returns:
        result: The moderation result (mocked).
    """
    # Placeholder for moderation logic
    # Replace this with actual model prediction logic
    result = 'acceptable'  # Mock result for illustration purposes
    return result



def prepare_image_from_pil(pil_img):
    # Convert the PIL image to grayscale if it isn't already
    pil_img = pil_img.convert('L')  # 'L' mode is for grayscale

    # Resize the image to the required size (30x30)
    pil_img = pil_img.resize((30, 30))

    # Convert the image to a NumPy array
    img_array = np.array(pil_img)

    # Normalize the image to [0, 1] range
    img_array = img_array / 255.0

    # Reshape to match the input shape required by Conv1D: (1, 30, 1)
    img_array = np.expand_dims(img_array, axis=-1)  # Add the channel dimension
    img_array = np.expand_dims(img_array, axis=0)   # Add the batch dimension

    return img_array

from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import streamlit as st

# Function to prepare the image from PIL
def prepare_image_from_pil(pil_img):
    # Check if the input is a numpy array
    if isinstance(pil_img, np.ndarray):
        # Convert the NumPy array to a PIL Image
        pil_img = Image.fromarray(pil_img)

    # Convert to grayscale
    pil_img = pil_img.convert('L')  # 'L' mode is for grayscale

    # Resize the image if necessary (e.g., to 30x30)
    pil_img = pil_img.resize((30, 30))

    # Convert the image to an array
    img_array = np.array(pil_img)

    # Flatten the image from 2D to 1D (30 features)
    img_array = np.mean(img_array, axis=1)  # Taking mean over rows to reduce 2D to 1D

    # Normalize the image to [0, 1] range
    img_array = img_array / 255.0

    # Reshape to match the input shape required by Conv1D: (1, 30, 1) where 1 is the batch size
    img_array = np.expand_dims(img_array, axis=-1)  # Add the channel dimension
    img_array = np.expand_dims(img_array, axis=0)   # Add the batch dimension

    return img_array

# Function to predict cancer
def predict_cancer_from_pil(pil_img):
    model = load_model('breast_cancer_model.keras')
    processed_image = prepare_image_from_pil(pil_img)

    # Get model prediction (sigmoid output gives probabilities)
    prediction = model.predict(processed_image)

    # Extract the scalar value from the array
    prediction = prediction[0][0]

    return prediction

def main():
    st.title("Automated Pectoral Muscle Removal & Cancer Detection")  # Title of the application
    
    # Instructions for users
    with st.expander("Click here for instructions on how to use the application"):
        st.markdown("""\
        ### Instructions
        1. **Upload a mammogram image**: Select a mammogram file in PNG, JPG, or JPEG format.
        2. The image will undergo several automatic processing steps:
            - **Enhancement**: To improve the contrast and clarity.
            - **Pectoral Muscle Removal**: This step will automatically detect and remove the pectoral muscle region from the image.
            - **Dense Tissue Detection**: The app will analyze the image for regions with dense breast tissue, which are of interest for diagnosis.
            - **Contours**: The app will outline dense and high-intensity regions with colored contours.
        3. Scroll through the results to see each processing step.
        """)
    
    # File uploader for mammogram images
    image_file = st.file_uploader("Upload a mammogram image...", type=["png", "jpg", "jpeg"])
    
    if image_file is not None:
        # Read the uploaded image
        original_image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        # st.image(original_image, caption='Original Image', use_column_width=True)  # Display the original image

        # Preprocessing steps
        preprocessed_image = preprocess_image(original_image)  # Preprocess the image for analysis
        
        num_labels, labels = find_connected_components(preprocessed_image)  # Find connected components in the image
        breast_mask = remove_non_breast_areas(preprocessed_image, labels, num_labels)  # Remove non-breast areas

        enhanced_image = enhance_image(original_image)  # Enhance the original image
        multi_otsu_results = apply_multi_otsu(preprocessed_image)  # Apply Multi-Otsu thresholding
        seed_pixel, seed_value = select_seed_pixel(multi_otsu_results)  # Select seed pixel for segmentation

        # Remove pectoral muscle and extract breast area
        pectoral_muscle_removed, breast_area = remove_pectoral_muscle(original_image, multi_otsu_results, seed_pixel)
        dense_region_image, dense_region_mask = extract_dense_regions(multi_otsu_results)  # Extract dense regions
        contour_image = draw_dense_region_contours(dense_region_image, dense_region_mask)  # Draw contours on the dense region image
        
        # Find the highest dense region
        highest_dense_image, _ = find_highest_dense_region(preprocessed_image)  # Capture both thresholded image and highest dense image
        
        # Create columns to display processed images side by side
        st.subheader("Processed Images")  # Add a subheader
# st.image(original_image, caption='Original Image', use_column_width=True)  # Display the original image
        col1, col2, col3 = st.columns(3)  # Create three columns

        with col1:
            st.image(original_image, caption='Original Image', use_column_width=True)

        with col2:
            st.image(enhanced_image, caption='Enhanced Image', use_column_width=True)
        
        with col3:
            st.image(pectoral_muscle_removed, caption='Pectoral Muscle Removed', use_column_width=True)
   
        
        # Create another row for dense regions and contours
        col4, col5 = st.columns(2)  # Create two more columns

        # Show highest dense region image side by side
        with col4:
            st.image(breast_area, caption='Breast Area', use_column_width=True) 
            
        with col5:
            st.image(highest_dense_image, caption="Highest Dense Region", use_column_width=True)

        # Perform moderation with loading animation
        with st.spinner("Processing your request..."):
            moderation_result = moderate_image(original_image)  # Call moderation function
        
        st.success("Moderation complete!")  # Display success message
        st.write(f'Moderation Result: {moderation_result}')  # Show the moderation result
                        
            # Make prediction
        prediction = predict_cancer_from_pil(breast_area)

            # Display the prediction result
        if prediction >= 0.5:
            st.success(f"The mammogram is classified as Malignant (Cancer) with a confidence of {prediction * 100:.2f}%.")
        else:
            st.success(f"The mammogram is classified as Benign (No Cancer) with a confidence of {(1 - prediction) * 100:.2f}%.")
                
if __name__ == "__main__":
    main()  # Run the main function     also show the highest dense region  
