# Experiment 02: Image Resizing and Interpolation

## 1. Objective

The objective of this experiment is to understand traditional image resizing and interpolation techniques and compare their reconstruction quality.

Three interpolation methods are evaluated:

1. Nearest Neighbour
2. Bilinear
3. Bicubic

The experiment establishes a traditional image-upscaling baseline before implementing deep learning-based super-resolution models.

## 2. Experimental Setup

A synthetic high-resolution image was generated using OpenCV.

| Parameter | Value |
|---|---|
| Original image dimensions | 256 × 256 |
| Low-resolution dimensions | 64 × 64 |
| Number of color channels | 3 |
| Image data type | uint8 |
| Downsampling method | INTER_AREA |
| Upscaling methods | Nearest, Bilinear, Bicubic |

The high-resolution image was downsampled from 256 × 256 to 64 × 64 and then upscaled back to 256 × 256 using each interpolation method.

The scale factor was 4× in each spatial dimension.

## 3. Interpolation Methods

### 3.1 Nearest Neighbour

Nearest Neighbour interpolation assigns the value of the nearest source pixel to the destination pixel.

- Computationally simple.
- Preserves sharp pixel boundaries.
- Can produce blocky or pixelated results.

OpenCV method: `cv2.INTER_NEAREST`

### 3.2 Bilinear Interpolation

Bilinear interpolation estimates a destination pixel using a weighted average of the four nearest source pixels.

- Produces smoother transitions than nearest neighbour.
- Reduces visible blockiness.
- Can soften edges and fine details.

OpenCV method: `cv2.INTER_LINEAR`

### 3.3 Bicubic Interpolation

Bicubic interpolation uses a 4 × 4 neighbourhood of source pixels and cubic interpolation to estimate destination pixel values.

- Uses more neighbouring information than bilinear interpolation.
- Often produces smoother and sharper-looking results.
- Requires more computation than nearest neighbour and bilinear interpolation.

OpenCV method: `cv2.INTER_CUBIC`

## 4. Experimental Pipeline

1. Generate a synthetic 256 × 256 high-resolution image.
2. Downsample it to 64 × 64 using area interpolation.
3. Upscale the low-resolution image to 256 × 256 using three methods.
4. Compare each reconstructed image against the original high-resolution image.
5. Calculate PSNR and SSIM.
6. Save the images and comparison visualization.

## 5. Evaluation Metrics

### Peak Signal-to-Noise Ratio (PSNR)

PSNR measures pixel-level reconstruction error using mean squared error.

Higher PSNR generally indicates lower pixel-level error.

### Structural Similarity Index (SSIM)

SSIM compares structural similarity between two images, considering luminance, contrast and structure.

Values closer to 1 indicate greater structural similarity.

## 6. Experimental Results

| Interpolation Method | PSNR (dB) | SSIM |
|---|---:|---:|
| Nearest Neighbour | 23.2652 | 0.8642 |
| Bilinear | 23.8626 | 0.8484 |
| Bicubic | 24.3040 | 0.8531 |

## 7. Observations

1. Bicubic interpolation achieved the highest PSNR of 24.3040 dB.
2. Nearest Neighbour achieved the highest SSIM of 0.8642.
3. Bilinear interpolation achieved a PSNR of 23.8626 dB and SSIM of 0.8484.
4. The method with the highest PSNR did not achieve the highest SSIM.
5. Interpolation increases image dimensions but does not reliably recover information lost during downsampling.

These results apply to the synthetic image and degradation setup used in this experiment. They should not be treated as universal rankings of interpolation methods.

## 8. Limitations

- The experiment uses a synthetic image rather than real surveillance footage.
- Only one image and one downsampling configuration were evaluated.
- PSNR and SSIM do not fully represent human perceptual quality.
- Traditional interpolation does not learn semantic or contextual image features.
- Upscaled images should not be interpreted as containing verified recovery of missing surveillance details.

## 9. Output Files

Generated files are stored locally in:

`outputs/images/experiment_02/`

- `original_hr.png`
- `low_resolution.png`
- `nearest.png`
- `bilinear.png`
- `bicubic.png`
- `interpolation_comparison.png`

## 10. Conclusion

This experiment established a traditional interpolation baseline for the image super-resolution project.

Bicubic achieved the highest PSNR, while Nearest Neighbour achieved the highest SSIM in this particular experiment.

These findings provide a baseline for future comparison with CNN-based and GAN-based super-resolution architectures.

## 11. Tools and Libraries

- Python
- OpenCV
- NumPy
- Matplotlib
- scikit-image