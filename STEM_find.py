import matplotlib.pyplot as plt
import machinevisiontoolbox as mvt
import numpy as np
import cv2 as cv 

def locate_stem(img: mvt.Image):
    """Locate the stem in an image by detecting brown regions and finding blobs."""
    
    # Step 1: reduce noise
    img_smooth = img.smooth(sigma=0.2)
    img_np = img_smooth.A  

    # Step 2: Convert to HSV color space
    hsv_img = cv.cvtColor(img_np, cv.COLOR_RGB2HSV)
    
    # Step 3: Define the color range for brown detection
    lower_brown = np.array([5, 50, 20])  
    upper_brown = np.array([30, 255, 200])
    
    # Step 4: Create a binary mask for the brown color
    mask_brown = cv.inRange(hsv_img, lower_brown, upper_brown)

    # Step 5: reduce noise
    kernel = np.ones((5, 5), np.uint8)
    mask_brown = cv.erode(mask_brown, kernel, iterations=1)
    mask_brown = cv.dilate(mask_brown, kernel, iterations=2)

    # Test-debug: Show the binary mask
    plt.figure(figsize=(6, 6))
    plt.imshow(mask_brown, cmap="gray")
    plt.title("Processed Binary Mask of Brown Regions")
    plt.show()

    # Test-debug: Check if the mask has non-zero pixels
    nonzero_pixels = np.count_nonzero(mask_brown)
    print(f" Non-zero pixels in the mask: {nonzero_pixels}")

    # If the mask is empty, return an empty list
    if nonzero_pixels == 0:
        print(" No pixels detected in the mask. Adjust the HSV thresholds.")
        return []

    # Step 6: Convert to an mvt.Image object for blob detection
    masked_image_obj = mvt.Image(mask_brown)

    try:
        # Step 7: Detect blobs without setting min/max area (manual filtering later)
        blobs = masked_image_obj.blobs()

        # Step 8: Manually filter blobs by area (ignore very small or large blobs)
        valid_blobs = [blob for blob in blobs if 50 < blob.area < 50000 and blob.moments[0] > 0]

        # Test-debug: Print number of blobs detected
        print(f"Number of valid blobs detected: {len(valid_blobs)}")

        # Step 9: Check if any valid blobs were found
        if len(valid_blobs) == 0:
            print(" No valid blobs detected. Check the binary mask segmentation.")
            return []

        # Step 10: Gather the centers of the detected blobs
        results = [blob.centroid for blob in valid_blobs]
        return results

    except Exception as e:
        print(f" Blob detection error: {e}")
        return []

# Load image
img_path = 'C:/Users/natyh/OneDrive - Queensland University of Technology/Documentos/SIMAC/STEM/bush_pic1.jpg' # Change to desired path
img = mvt.Image.Read(img_path)

# Call the function
results = locate_stem(img)

# Print the detected centroids
print("Detected stems and their properties:")
for result in results:
    print(f"Centroid: {result}")

# Step 11: Draw the detected centers on the original image
img_np = img.A  
for centroid in results:
    cv.circle(img_np, (int(centroid[0]), int(centroid[1])), 5, (255, 0, 0), -1)  # Draw a red dot at the center

# Step 12: Show the image with the detected blobs
img_np_bgr = cv.cvtColor(img_np, cv.COLOR_RGB2BGR)
plt.figure(figsize=(6, 6))
plt.imshow(cv.cvtColor(img_np_bgr, cv.COLOR_BGR2RGB))
plt.title("Detected Stems")
plt.show()
