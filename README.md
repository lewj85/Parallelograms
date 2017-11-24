# Parallelograms

## Detects parallelograms in any image

Converts image to grayscale
Resizes image for faster processing and thinner edges
Mean filter
Gaussian/Laplacian/LoG filter
Post-filter thresholding for noise reduction
Edge detection using Sobel/Canny mask
Nonmaxima suppression for thin edges
Edge thresholding for strong edges
Hough transform to find all straight lines
Filter lines by strength/length
Find longest and shortest lines (optional)
Find intersection points
Filter intersection points based on distance and angle 
Draw lines between remaining sets of intersection points
