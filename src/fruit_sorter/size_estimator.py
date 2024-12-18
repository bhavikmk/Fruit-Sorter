import cv2
import numpy as np

global pixel_to_cm_ratio

pixel_to_cm_ratio =  0.035  

def estimate_mango_size(image_path):
    """
    Estimate the size of a mango in an image based on its contours.
    
    Parameters:
        image_path (str): Path to the input image of the mango.
        
    Returns:
        dict: Contains the estimated length and width of the mango in cm.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not load the image. Check the path.")
    
    # resized_image = cv2.resize(image, (640, 480))
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Perform edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by area and pick the largest one (assuming it's the mango)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    if len(contours) == 0:
        raise ValueError("No contours detected.")
    mango_contour = contours[0]
    
    # Calculate bounding rectangle
    x, y, w, h = cv2.boundingRect(mango_contour)
    
    # Highlight the detected mango in the original image
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(image, "Mango", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Estimate the size in pixels
    mango_length_px = h
    mango_width_px = w
    
    # Convert mango dimensions to cm
    mango_length_cm = mango_length_px * pixel_to_cm_ratio
    mango_width_cm = mango_width_px * pixel_to_cm_ratio
    
    # Display the results
    cv2.putText(
        image,
        f"Length: {mango_length_cm:.2f} cm, Width: {mango_width_cm:.2f} cm",
        (5, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 100, 0),
        2
    )
    cv2.imshow("Mango Size Estimation", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Return the dimensions
    return {"length_cm": mango_length_cm, "width_cm": mango_width_cm}


# Example usage
image_path = "dataset/Stage_0 (Unripe)/Training/IMG20200713141551.jpg"  
size = estimate_mango_size(image_path)
print("Estimated Mango Size:", size)
