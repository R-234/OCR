

import streamlit as st
import os
import time
import cv2
import pytesseract
import numpy as np
from PIL import Image
import re
from zipfile import ZipFile
import shutil
import concurrent.futures  # Multi-threading support

# Define directories
IMAGE_FOLDER = "upload_images"
OUTPUT_FOLDER = "renamed_images"
FAILED_FOLDER = "failed_images"
ZIP_SUCCESS = "processed_results.zip"
ZIP_FAILED = "failed_results.zip"

# Ensure directories exist
for folder in [IMAGE_FOLDER, OUTPUT_FOLDER, FAILED_FOLDER]:
    os.makedirs(folder, exist_ok=True)

ROI_COORDS = (50, 70, 450, 350)  # Fixed Region of Interest


def resize_image(image, max_size=800):
    """Resize image to a maximum width/height while maintaining aspect ratio."""
    h, w = image.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return image


def preprocess_image(image):
    """Optimized preprocessing: Resize, grayscale, threshold."""
    try:
        image = resize_image(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # Faster than bilateralFilter
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Faster than adaptiveThreshold
        return thresh
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return None


def extract_text(image):
    """Extract numbers from a fixed region of interest (ROI)."""
    try:
        x1, y1, x2, y2 = ROI_COORDS
        roi = image[y1:y2, x1:x2]
        extracted_text = pytesseract.image_to_string(roi, config='--psm 6 --oem 1').strip()  # Use OEM 1 for better speed
        numbers_only = re.sub(r'\D', '', extracted_text)

        if numbers_only and 4 <= len(numbers_only) <= 5:
            return numbers_only.zfill(4)
        return None
    except Exception as e:
        print(f"Error in text extraction: {e}")
        return None


def process_single_image(image_file):
    """Process a single image and return the result."""
    image_path = os.path.join(IMAGE_FOLDER, image_file)
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error loading image: {image_file}")
        return image_file, None  # Failed case

    processed_image = preprocess_image(image)
    if processed_image is None:
        return image_file, None  # Failed case

    extracted_number = extract_text(processed_image)
    if not extracted_number:
        return image_file, None  # Failed case

    new_filename = f"{extracted_number}.jpg"
    output_path = os.path.join(OUTPUT_FOLDER, new_filename)
    success = cv2.imwrite(output_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

    return (None, new_filename) if success else (image_file, None)  # Success or Failed case


def process_images_parallel(image_files):
    """Process multiple images using multi-threading."""
    renamed_files, failed_files = [], []
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(process_single_image, image_files)

    for failed, renamed in results:
        if failed:
            failed_files.append(failed)
            shutil.move(os.path.join(IMAGE_FOLDER, failed), os.path.join(FAILED_FOLDER, failed))
        if renamed:
            renamed_files.append(renamed)

    return renamed_files, failed_files


def create_zip(folder, zip_name):
    """Create a ZIP file of all images in a folder."""
    zip_path = zip_name
    with ZipFile(zip_path, 'w') as zipf:
        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)
            zipf.write(file_path, arcname=file)
    return zip_path


def cleanup_folders():
    """Delete processed and failed images after zipping."""
    for folder in [IMAGE_FOLDER, OUTPUT_FOLDER, FAILED_FOLDER]:
        for file in os.listdir(folder):
            os.remove(os.path.join(folder, file))

def cleanup_temp_files():
    folders = ["upload_images", "renamed_images", "failed_images"]
    for folder in folders:
        shutil.rmtree(folder, ignore_errors=True)
        os.makedirs(folder, exist_ok=True)

    # Delete ZIP files
    for zip_file in ["processed_results.zip", "failed_results.zip"]:
        if os.path.exists(zip_file):
            os.remove(zip_file)

    
  
def process_images():
    # OCR processing logic here
    print("OCR processing completed.")

process_images()
cleanup_temp_files()  # Delete extra files after processing
print("Temporary files deleted.")






def process_and_zip(uploaded_files):
    """Processes images using OCR and returns ZIP file paths."""
    image_files = []
    for file in uploaded_files:
        if file.size > 5 * 1024 * 1024:
            st.warning(f"Skipping {file.name} (too large).")
            continue
        
        file_path = os.path.join(IMAGE_FOLDER, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        image_files.append(file.name)

    renamed_files, failed_files = process_images_parallel(image_files)  # Use parallel processing

    success_zip = create_zip(OUTPUT_FOLDER, ZIP_SUCCESS) if renamed_files else None
    failed_zip = create_zip(FAILED_FOLDER, ZIP_FAILED) if failed_files else None

    cleanup_folders()
    return success_zip, failed_zip


# Streamlit UI
st.markdown(
    """
    <style>
    @keyframes scrollText {
        0% { transform: translateX(100%); }
        100% { transform: translateX(-100%); }
    }
    .scrolling-text {
        white-space: nowrap;
        overflow: hidden;
        position: relative;
        width: 100%;
        font-size: 20px;
        font-weight: bold;
        color: #0066cc;
    }
    .scrolling-text span {
        display: inline-block;
        animation: scrollText 10s linear infinite;
    }
    .title {
        font-size: 30px;
        font-weight: bold;
        color: #ff5733;
        text-align: center;
    }
    </style>
    <div class='scrolling-text'><span>Blacklead Infratech PVT LTD</span></div>
    """,
    unsafe_allow_html=True
)

st.markdown("<div class='title'>OCR Model for BIPL</div>", unsafe_allow_html=True)

uploaded_files = st.file_uploader("Upload Images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if st.button("Process Images"):
    if uploaded_files:
        with st.spinner("Processing images..."):
            success_zip, failed_zip = process_and_zip(uploaded_files)

        st.success("Processing Completed!")

        if success_zip:
            with open(success_zip, "rb") as file:
                st.download_button("Download Processed Images (ZIP)", file, ZIP_SUCCESS, "application/zip")
        if failed_zip:
            with open(failed_zip, "rb") as file:
                st.download_button("Download Failed Images (ZIP)", file, ZIP_FAILED, "application/zip")
    else:
        st.warning("Please upload images before processing.")








