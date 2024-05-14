import cv2
import pytesseract
import numpy as np
import re
import os
import time
from datetime import datetime


# Function to preprocess the image
def preprocess_image(image):
    # Apply unsharp masking to enhance edges (sharpening)
    blurred = cv2.GaussianBlur(image, (0, 0), 3)
    sharpened_image = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)

    # Convert the image to grayscale
    gray = cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to get a binary image
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Apply morphological operations to clean up the image
    kernel = np.ones((1, 1), np.uint8)
    processed_image = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    return processed_image


# Function to extract the Aadhar number using regex
def extract_aadhar_number(text):
    pattern = r'\b\d{4}\s?\d{4}\s?\d{4}\b'
    match = re.search(pattern, text)
    if match:
        return match.group().replace(" ", "")
    return "No Aadhar Number Found"


# Function to save the Aadhar number and image to files
def save_aadhar_info(image, full_image, aadhar_number):
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Generate unique filenames using timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = os.path.join(output_dir, f"captured_image_{timestamp}.jpg")
    full_image_path = os.path.join(output_dir, f"full_aadhar_image_{timestamp}.jpg")
    text_path = os.path.join(output_dir, f"aadhar_number_{timestamp}.txt")

    # Save the images
    cv2.imwrite(image_path, image)
    cv2.imwrite(full_image_path, full_image)

    # Save the Aadhar number to a text file
    with open(text_path, 'w') as file:
        file.write(f"Aadhar Number: {aadhar_number}\n")


# Function to capture a frame from the camera
def capture_frame(cap):
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to capture frame.")
        return None
    return frame


# Capture image from camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Unable to open camera.")
    exit()

# Define the rectangle parameters
rect_top_left = (100, 100)
rect_bottom_right = (540, 280)

# Capture the full Aadhar card image
print("Capturing full Aadhar card image...")
full_aadhar_image = capture_frame(cap)
if full_aadhar_image is None:
    cap.release()
    exit()

# Capture the region of interest (ROI) within the rectangle
print("Capturing Aadhar number ROI...")
capture_successful = False
start_time = time.time()
while (time.time() - start_time) < 4:  # Capture frames for 4 seconds
    frame = capture_frame(cap)
    if frame is None:
        break

    # Draw the rectangle on the frame
    cv2.rectangle(frame, rect_top_left, rect_bottom_right, (0, 255, 0), 2)

    # Show the live video feed
    cv2.imshow('Camera Preview', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit early
        break

    captured_frame = frame
    capture_successful = True

# Release the camera
cap.release()
cv2.destroyAllWindows()

if not capture_successful:
    print("Error: Failed to capture frame.")
    exit()

# Extract the region of interest (ROI) within the rectangle
roi = captured_frame[rect_top_left[1]:rect_bottom_right[1], rect_top_left[0]:rect_bottom_right[0]]

# Preprocess the captured image
processed_image = preprocess_image(roi)

# Use Tesseract to extract text from the preprocessed image
custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789'  # Only recognize numbers
text = pytesseract.image_to_string(processed_image, config=custom_config, lang='eng')

# Print the recognized text for debugging
print("Recognized Text:", text)

# Extract the Aadhar number from the recognized text
aadhar_number = extract_aadhar_number(text)

# Print the extracted Aadhar number
print("Aadhar Number:", aadhar_number)

# Save the Aadhar number and images to files
save_aadhar_info(processed_image, full_aadhar_image, aadhar_number)
