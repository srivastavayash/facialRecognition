import cv2
import pytesseract
import numpy as np
import re
import time
import os
from datetime import datetime

# Function to preprocess the image
def preprocess_image(image):
    blurred = cv2.GaussianBlur(image, (0, 0), 3)
    sharpened_image = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
    gray = cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    kernel = np.ones((1, 1), np.uint8)
    processed_image = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return processed_image

# Function to extract the Aadhar number using regex
def extract_aadhar_number(text):
    pattern = r'\b\d{4}\s?\d{4}\s?\d{4}\b'
    match = re.search(pattern, text)
    if match:
        return match.group().replace(" ", "")
    return None

# Function to create and save the combined image with bookmark and date-time watermark
def create_combined_image(aadhar_number, full_image, party_name):
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Generate unique filename using timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    combined_image_path = os.path.join(output_dir, f"combined_image_{timestamp}.jpg")

    # Create an empty image for the bookmark with white background
    bookmark_height = 100
    bookmark_width = full_image.shape[1]
    bookmark_image = np.ones((bookmark_height, bookmark_width, 3), dtype=np.uint8) * 255

    # Add the party name text to the bookmark image
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness =3
    text_size = cv2.getTextSize(party_name, font, font_scale, font_thickness)[0]
    text_x = (bookmark_width - text_size[0]) // 1
    text_y = (bookmark_height + text_size[1]) // 4
    overlay = bookmark_image.copy()
    cv2.putText(overlay, party_name, (text_x, text_y), font, font_scale, (0, 0, 255), font_thickness)
    alpha = 0.3  # Opacity factor
    cv2.addWeighted(overlay, alpha, bookmark_image, 1 - alpha, 0, bookmark_image)

    # Convert the grayscale full image to a 3-channel image
    full_image_colored = cv2.cvtColor(full_image, cv2.COLOR_GRAY2BGR)

    # Combine the bookmark image and the full Aadhar image
    combined_image = np.vstack((bookmark_image, full_image_colored))

    # Add the date-time watermark diagonally
    watermark_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    combined_text = f"{aadhar_number} {watermark_text}"
    step = 50  # Adjust step size to prevent overlapping
    for i in range(0, combined_image.shape[0], step):
        pos_x = i
        pos_y = i+10
        if pos_x < combined_image.shape[1] and pos_y < combined_image.shape[0]:
            cv2.putText(combined_image, combined_text, (pos_x, pos_y), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Save the combined image
    cv2.imwrite(combined_image_path, combined_image)
    return combined_image_path

# Function to resize and convert the image to grayscale
def resize_and_grayscale(image, max_inch=2, dpi=300):
    max_pixels = int(max_inch * dpi)
    height, width = image.shape[:2]
    max_dim = max(height, width)
    scale = max_pixels / max_dim
    new_size = (int(width * scale), int(height * scale))
    resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    grayscale_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    return grayscale_image

# Function to capture a frame from the camera
def capture_frame(cap):
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to capture frame.")
        return None
    return frame

# Function to capture the ROI for Aadhar number
def capture_roi(cap, rect_top_left, rect_bottom_right):
    print("Capturing Aadhar number ROI...")
    capture_successful = False
    start_time = time.time()
    captured_frame = None
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

    cv2.destroyAllWindows()
    return captured_frame if capture_successful else None

# Main function to capture images and extract Aadhar number
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to open camera.")
        return

    # Define the rectangle parameters
    rect_top_left = (100, 100)
    rect_bottom_right = (540, 280)

    # Capture the region of interest (ROI) within the rectangle
    captured_frame = capture_roi(cap, rect_top_left, rect_bottom_right)
    if captured_frame is None:
        print("Error: Failed to capture frame.")
        return

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

    # Check if Aadhar number is found
    if aadhar_number is None:
        print("Aadhar number not found. Try again.")
        return

    # Print the extracted Aadhar number
    print("Aadhar Number:", aadhar_number)

    # Prompt the user to adjust the Aadhar card for full image capture
    print("Please adjust the Aadhar card for full image capture. Press 'c' to capture or 'q' to quit.")

    full_aadhar_image = None
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture frame.")
            continue

        # Show the live video feed
        cv2.imshow('Full Aadhar Capture', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            full_aadhar_image = frame
            break
        elif key == ord('q'):
            print("Quitting...")
            return

    cv2.destroyAllWindows()
    cap.release()

    # Resize and convert the captured full image to grayscale
    full_aadhar_image = resize_and_grayscale(full_aadhar_image)

    # Read the party name from the text file
    with open("party_name.txt", "r") as file:
        party_name = file.read().strip()

    if party_name == "A":
        party_name = "Party A"
    elif party_name == "B":
        party_name = "Party B"
    elif party_name == "C":
        party_name = "Party C"
    else:
        print("Invalid party name in text file.")
        return

    # Create the combined image with bookmark and date-time watermark
    combined_image_path = create_combined_image(aadhar_number, full_aadhar_image, party_name)
    print(f"Combined image saved at: {combined_image_path}")

if __name__ == "__main__":
    main()
