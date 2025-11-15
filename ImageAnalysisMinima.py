# Multiple Slit Diffraction Image Analysis Code - Central 8 Minima Finder
# Gunner Imel
# Last Updated: 11/10/25

import cv2
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# read image
image_path = "C:/Users/gunne/Desktop/IU and Purdue classes/" \
"PHYS 401 Advanced Physics Lab II/Multiple Slit Diffraction/SlitDiffraction_0.08_0.500_N2_1.jpg"
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    raise ValueError("Image not found or path is incorrect.")

# Get middle row of pixels
rows, cols = img.shape
middle_row_index = rows // 2
intensity_row = img[middle_row_index, :]

# Normalize intensity
intensity_normalized = intensity_row / np.max(intensity_row)

# Convert horizontal pixels to mm
scale_factor = cols / 40.0  # pixels per mm
x_mm = np.arange(cols) / scale_factor
x_mm = x_mm - 20  # shift x-axis

print("Pixel resolution: ", scale_factor)

# Find Minima
# Invert the intensity to detect minima as peaks in the inverted data
inverted_intensity = -intensity_normalized

# Detect minima
minima, properties = find_peaks(inverted_intensity, height=-1.0, distance=60)

# Optional: refine minima centers (for flat minima)
refined_minima = []
for p in minima:
    val = intensity_normalized[p]
    tol = 0.1
    left = p
    right = p

    # Expand left
    while left > 0 and abs(intensity_normalized[left - 1] - val) < tol:
        left -= 1
    # Expand right
    while right < len(intensity_normalized) - 1 and abs(intensity_normalized[right + 1] - val) < tol:
        right += 1

    # Center of flat region if found
    if right > left:
        center = (left + right) // 2
        refined_minima.append(center)
    else:
        refined_minima.append(p)

refined_minima = np.array(refined_minima, dtype=int)

# Find the minimum closest to the center
center_idx = np.argmin(np.abs(x_mm[refined_minima]))

# Sort minima by distance from center
sorted_by_distance = refined_minima[np.argsort(np.abs(x_mm[refined_minima]))]

# Select the 8 closest minima
central_8_minima = np.sort(sorted_by_distance[:8])

print("Central 8 minima (positions in mm):", x_mm[central_8_minima])
print("Central 8 minima (intensity values):", intensity_normalized[central_8_minima])

# Experiment Setup Details
laser_loc = 829      
slit_loc = 604              
screen_loc = 91 
wavelength = 632.8e-6

D = slit_loc - screen_loc
n = np.array([-4,-3,-2,-1,1,2,3,4])
a_vals = []

# Calculate and return a
i = 0
for x in x_mm[central_8_minima]:
    a = n[i] * wavelength * D / x
    a_vals.append(a)
    i = i + 1

print("a values: ", a_vals)

# Plot results
plt.figure(figsize=(8, 4))
plt.plot(x_mm, intensity_normalized, color='black', linewidth=1.5)
plt.plot(x_mm[central_8_minima], intensity_normalized[central_8_minima], "bo", label="Minima")
plt.title("Normalized Intensity of Diffraction Pattern")
plt.xlabel("Position (mm)")
plt.ylabel("I / I0")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()