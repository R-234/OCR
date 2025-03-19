import os
import cv2
import pytesseract
import numpy as np
import zipfile
from PIL import Image
import re
from collections import defaultdict
import shutil

# Set paths
IMAGE_FOLDER = "input_images"   # Folder containing input images
OUTPUT_FOLDER = "renamed_images"  # Folder where renamed images will be saved
FAILED_FOLDER = "failed_images"   # Folder for failed images

# Ensure output directories exist
os.makedirs(IMAGE_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(FAILED_FOLDER, exist_ok=True)

# Define fixed coordinates (ROI) for extracting numbers (adjust as needed)
ROI_COORDS = (50, 70, 450, 350)  # (x1, y1, x2, y2)

def preprocess_image(image):
    """Preprocess image for better OCR recognition with noise reduction."""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        bilateral = cv2.bilateralFilter(gray, 9, 75, 75)  # Reduce noise while keeping edges
        adaptive_thresh = cv2.adaptiveThreshold(
            bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
        return cleaned
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return None

def extract_text(image):
    """Extract numbers from a fixed region of interest (ROI) with enhanced filtering."""
    try:
        x1, y1, x2, y2 = ROI_COORDS
        roi = image[y1:y2, x1:x2]  # Crop the region of interest
        roi_pil = Image.fromarray(roi)
        extracted_text = pytesseract.image_to_string(roi_pil, config='--psm 6').strip()
        numbers_only = re.sub(r'\D', '', extracted_text)  # Extract only digits

        print(f"Extracted Text: {extracted_text} -> Numbers Only: {numbers_only}")  # Debugging line

        if numbers_only and 4 <= len(numbers_only) <= 5:
            return numbers_only.zfill(4)  # Ensure filename format "0000.jpg" or "00000.jpg"
        return None
    except Exception as e:
        print(f"Error in text extraction: {e}")
        return None


def rename_images(image_files):
    """Process and rename images based on extracted text."""
    existing_names = defaultdict(int)  # Track duplicate file names
    failed_files = []
    renamed_files = []

    for image_file in image_files:
        image_path = os.path.join(IMAGE_FOLDER, image_file)

        if not os.path.exists(image_path):
            print(f"File not found: {image_path}")
            continue  # Skip missing files

        image = cv2.imread(image_path)
        if image is None:
            print(f"Error loading image: {image_file}")
            failed_files.append(image_file)
            shutil.move(image_path, os.path.join(FAILED_FOLDER, image_file))
            continue

        print(f"Processing: {image_file}")
        processed_image = preprocess_image(image)
        
        if processed_image is None:
            print(f"Skipping {image_file} due to preprocessing error.")
            failed_files.append(image_file)
            shutil.move(image_path, os.path.join(FAILED_FOLDER, image_file))
            continue

        extracted_number = extract_text(processed_image)

        if not extracted_number:
            print(f"No valid number found in {image_file}. Moving to failed folder.")
            failed_files.append(image_file)
            shutil.move(image_path, os.path.join(FAILED_FOLDER, image_file))
            continue

        # Ensure unique filenames
        new_filename = f"{extracted_number}.jpg"
        if existing_names[extracted_number]:
            new_filename = f"{extracted_number}_{existing_names[extracted_number]}.jpg"
        
        existing_names[extracted_number] += 1
        output_path = os.path.join(OUTPUT_FOLDER, new_filename)
        
        # Save renamed image
        success = cv2.imwrite(output_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if success:
            renamed_files.append(new_filename)
            print(f"Renamed & Saved: {new_filename}")
        else:
            print(f"Failed to save image: {image_file}")
            failed_files.append(image_file)
            shutil.move(image_path, os.path.join(FAILED_FOLDER, image_file))
    
    return renamed_files, failed_files

def create_zip(folder, zip_name):
    """Create a ZIP file of all images in a folder."""
    zip_path = os.path.join(folder, zip_name)
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)
            zipf.write(file_path, arcname=file)
    print(f"ZIP Created: {zip_path} -> Contains: {os.listdir(folder)}")  # Debugging line

def main():
    image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print("No images found for processing!")
        return

    renamed_files, failed_files = rename_images(image_files)

    # Create ZIPs
    if renamed_files:
        create_zip(OUTPUT_FOLDER, "renamed_images.zip")
    if failed_files:
        create_zip(FAILED_FOLDER, "failed_images.zip")
    
    print("All images processed successfully!")

if __name__ == "__main__":
    main()
    







# import os
# import cv2
# import pytesseract
# import numpy as np
# import zipfile
# from PIL import Image
# import re
# from collections import defaultdict
# import shutil

# # Set paths
# IMAGE_FOLDER = "input_images"   # Folder containing input images
# OUTPUT_FOLDER = "renamed_images"  # Folder where renamed images will be saved
# FAILED_FOLDER = "failed_images"   # Folder for failed images

# # Ensure output directories exist
# os.makedirs(IMAGE_FOLDER, exist_ok=True)
# os.makedirs(OUTPUT_FOLDER, exist_ok=True)
# os.makedirs(FAILED_FOLDER, exist_ok=True)

# def enhance_contrast(image):
#     """Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
#     lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
#     l, a, b = cv2.split(lab)
#     clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
#     cl = clahe.apply(l)
#     enhanced_lab = cv2.merge((cl, a, b))
#     return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

# def correct_skew(image):
#     """Detect and correct skew using Hough Transform."""
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     edges = cv2.Canny(gray, 50, 150, apertureSize=3)
#     lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
#     if lines is not None:
#         angle = np.mean([np.degrees(np.arctan2(np.sin(rho), np.cos(rho))) for rho, theta in lines[:, 0]])
#         if abs(angle) > 0.5:
#             (h, w) = image.shape[:2]
#             M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
#             return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
#     return image

# def preprocess_image(image):
#     """Advanced preprocessing for OCR recognition."""
#     image = enhance_contrast(image)
#     image = correct_skew(image)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     adaptive_thresh = cv2.adaptiveThreshold(
#         blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
#     )
#     kernel = np.ones((3, 3), np.uint8)
#     cleaned = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
#     return cleaned

# def extract_text(image):
#     """Extract numbers from dynamically detected ROI."""
#     try:
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         for cnt in contours:
#             x, y, w, h = cv2.boundingRect(cnt)
#             if 50 < w < 500 and 20 < h < 200:  # Filtering based on expected text size
#                 roi = image[y:y+h, x:x+w]
#                 extracted_text = pytesseract.image_to_string(roi, config='--psm 6').strip()
#                 numbers_only = re.sub(r'\D', '', extracted_text)
#                 if numbers_only and 4 <= len(numbers_only) <= 5:
#                     return numbers_only.zfill(4)
#         return None
#     except Exception as e:
#         print(f"Error in text extraction: {e}")
#         return None

# def rename_images(image_files):
#     """Process and rename images based on extracted text."""
#     existing_names = defaultdict(int)
#     failed_files = []
#     renamed_files = []

#     for image_file in image_files:
#         image_path = os.path.join(IMAGE_FOLDER, image_file)
#         image = cv2.imread(image_path)
#         if image is None:
#             failed_files.append(image_file)
#             shutil.move(image_path, os.path.join(FAILED_FOLDER, image_file))
#             continue

#         print(f"Processing: {image_file}")
#         processed_image = preprocess_image(image)
#         extracted_number = extract_text(processed_image)

#         if not extracted_number:
#             failed_files.append(image_file)
#             shutil.move(image_path, os.path.join(FAILED_FOLDER, image_file))
#             continue

#         new_filename = f"{extracted_number}.jpg"
#         if existing_names[extracted_number]:
#             new_filename = f"{extracted_number}_{existing_names[extracted_number]}.jpg"
#         existing_names[extracted_number] += 1
#         output_path = os.path.join(OUTPUT_FOLDER, new_filename)
        
#         if cv2.imwrite(output_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 90]):
#             renamed_files.append(new_filename)
#             print(f"Renamed & Saved: {new_filename}")
#         else:
#             failed_files.append(image_file)
#             shutil.move(image_path, os.path.join(FAILED_FOLDER, image_file))
    
#     return renamed_files, failed_files

# def create_zip(folder, zip_name):
#     """Create a ZIP file of all images in a folder."""
#     zip_path = os.path.join(folder, zip_name)
#     with zipfile.ZipFile(zip_path, 'w') as zipf:
#         for file in os.listdir(folder):
#             zipf.write(os.path.join(folder, file), arcname=file)
#     print(f"ZIP Created: {zip_path}")

# def main():
#     image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
#     if not image_files:
#         print("No images found for processing!")
#         return

#     renamed_files, failed_files = rename_images(image_files)
#     if renamed_files:
#         create_zip(OUTPUT_FOLDER, "renamed_images.zip")
#     if failed_files:
#         create_zip(FAILED_FOLDER, "failed_images.zip")
#     print("All images processed successfully!")

# if __name__ == "__main__":
#     main()

