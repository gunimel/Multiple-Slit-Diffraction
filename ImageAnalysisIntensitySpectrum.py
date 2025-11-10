# Multiple Slit Diffraction Image Analysis Code
# Gunner Imel
# Last Updated: 11/10/25

import cv2
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# Read image
image_path = "C:/Users/gunne/Desktop/IU and Purdue classes/" \
"PHYS 401 Advanced Physics Lab II/Multiple Slit Diffraction/SlitDiffraction_0.08_0.500_N2_2.jpg"
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    raise ValueError("Image not found or path is incorrect.")

# Extract central row
rows, cols = img.shape
middle_row_index = rows // 2
intensity_row = img[middle_row_index, :]

# Normalize intensity
intensity_normalized = intensity_row / np.max(intensity_row)

# Convert horizontal pixels to mm
scale_factor = cols / 40.0  # pixels per mm
x_mm = np.arange(cols) / scale_factor
x_mm = x_mm - 20  # shift x-axis

# Return pixel resolution
print("Pixel resolution: ", scale_factor)

# Find peaks
peaks, properties = find_peaks(intensity_normalized, height=0.1, distance=60)

# Center peaks if flat regions exist
refined_peaks = []
for p in peaks:
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
        refined_peaks.append(center)
    else:
        refined_peaks.append(p)

refined_peaks = np.array(refined_peaks, dtype=int)

# Filter out weak peaks
strong_peaks = [p for p in refined_peaks if intensity_normalized[p] >= 0.6]
strong_peaks = np.array(strong_peaks, dtype=int)

# Filter out duplicate peaks
strong_peaks = np.sort(np.unique(strong_peaks))

# Print results 
print("Filtered maxima (positions in mm):", x_mm[strong_peaks])
print("Filtered maxima (intensity values):", intensity_normalized[strong_peaks])

# Find beta
laser_loc = 829      
slit_loc = 604              
screen_loc = 91 
k = 2 * np.pi / (633e-6)    # wave number for wavelength 633 nm

d = laser_loc - screen_loc
n = 0
for x in x_mm[strong_peaks]:
    tantheta = x / d
    beta = np.sqrt(intensity_normalized[strong_peaks] **(-1)) 
    a = 2 * beta * d / (x * k) 
    n = n + 1

print("a values: ", a)

# Plot results
plt.figure(figsize=(8, 4))
plt.plot(x_mm, intensity_normalized, color='black', linewidth=1.5)
plt.plot(x_mm[strong_peaks], intensity_normalized[strong_peaks], "ro", label="Maxima")
plt.title("Normalized Intensity of Diffraction Pattern")
plt.xlabel("Position (mm)")
plt.ylabel("I / I0")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()