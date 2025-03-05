import os
import cv2
import easyocr

# Fix OpenMP conflict issue
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

# Define file paths
cascade_path = "numberplate_haarcade.xml"
image_path = "F:/PROJECT/ML/image.jpeg"

# Check if cascade file exists
if not os.path.exists(cascade_path):
    print(f"Error: Haarcascade file '{cascade_path}' not found!")
    exit()

# Check if image file exists
if not os.path.exists(image_path):
    print(f"Error: Image file '{image_path}' not found!")
    exit()

# Initialize cascade classifier
detector = cv2.CascadeClassifier(cascade_path)

# Validate the classifier is loaded properly
if detector.empty():
    print("Error: Failed to load Haarcascade classifier!")
    exit()

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Read image
img = cv2.imread(image_path)

# Convert image to grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect number plates
plates = detector.detectMultiScale(img_gray, scaleFactor=1.05, minNeighbors=7, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

if len(plates) == 0:
    print("No number plates detected!")
else:
    print(f"Detected {len(plates)} plate(s): {plates}")

# Process each detected plate
for (x, y, w, h) in plates:
    # Draw bounding box around the plate
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Crop the number plate
    plateROI = img_gray[y:y+h, x:x+w]
    cv2.imwrite("output.jpg", plateROI)

    # Detect text using EasyOCR
    text = reader.readtext(plateROI)

    if len(text) == 0:
        print("No text detected on plate!")
        continue

    # Extract and clean the detected text
    detected_text = text[0][1].strip('/\\| ')
    print(f"Detected Text: {detected_text}")

    # Draw the detected text on the image
    cv2.putText(img, detected_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

# Show the final output
cv2.imshow('Output', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
