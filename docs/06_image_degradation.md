# Experiment 03: Image Degradation Pipeline

## 1. Objective

The objective of this experiment is to implement a controlled image degradation pipeline that generates low-resolution images from high-resolution images.

The pipeline simulates common image-quality problems encountered in surveillance footage:

1. Gaussian blur
2. Gaussian noise
3. JPEG compression
4. Downsampling

This experiment establishes the foundation for generating paired HR-LR training data for future super-resolution models.

## 2. Experimental Setup

A synthetic high-resolution image was generated using OpenCV.

| Parameter | Value |
|---|---|
| Original image size | 256 × 256 |
| Degraded image size | 64 × 64 |
| Image channels | 3 |
| Data type | uint8 |
| Gaussian kernel | 5 × 5 |
| Gaussian sigma | 1.0 |
| Noise standard deviation | 10 |
| JPEG quality | 40 |
| Downsampling method | INTER_AREA |
| Downsampling scale | 4× |

A fixed random seed of 42 was used for reproducible Gaussian noise generation.

## 3. Degradation Techniques

### 3.1 Gaussian Blur

Gaussian blur simulates optical defocus and loss of fine image details.

Implementation:

`cv2.GaussianBlur(image, (5, 5), sigmaX=1.0)`

The kernel size controls the neighbourhood used for filtering, while sigma controls the Gaussian distribution's spread.

### 3.2 Gaussian Noise

Gaussian noise simulates random intensity variations that may occur due to image sensor noise and other acquisition conditions.

The noise model is:

`I_noisy = I_clean + epsilon`

where epsilon follows a Gaussian distribution with mean zero and standard deviation 10.

The resulting pixel values are clipped to the valid uint8 range of 0–255.

### 3.3 JPEG Compression

JPEG compression simulates lossy image compression artifacts.

The experiment uses JPEG quality 40.

Lower JPEG quality settings generally introduce stronger compression artifacts and loss of image details.

### 3.4 Downsampling

The image is reduced from 256 × 256 to 64 × 64 using OpenCV area interpolation.

Implementation:

`cv2.INTER_AREA`

The spatial downsampling factor is 4×.

## 4. Combined Degradation Pipeline

The combined pipeline applies the following operations sequentially:

1. Start with the original high-resolution image.
2. Apply Gaussian blur.
3. Add Gaussian noise.
4. Apply JPEG compression.
5. Downsample the compressed image to 64 × 64.

The resulting image is used as a synthetic low-resolution sample.

## 5. Experimental Results

The script successfully generated the following outputs:

| Output | Description |
|---|---|
| original_hr.png | Original high-resolution image |
| gaussian_blur.png | Image after Gaussian blur |
| gaussian_noise.png | Image with Gaussian noise |
| jpeg_compression.png | JPEG-compressed image |
| downsampled_lr.png | Downsampled low-resolution image |
| combined_degraded_lr.png | Image after combined degradation |

Verified image dimensions:

- Original HR: 256 × 256 × 3
- Downsampled LR: 64 × 64 × 3
- Combined LR: 64 × 64 × 3

## 6. Reproducibility

A NumPy random generator with seed 42 was used to generate Gaussian noise.

This allows the same noise pattern to be reproduced when the code and execution environment remain unchanged.

## 7. Limitations

- The experiment uses a synthetic image rather than real surveillance footage.
- Only one set of degradation parameters is evaluated.
- The degradation sequence is a controlled simulation and does not represent every camera pipeline.
- Real surveillance footage may contain motion blur, sensor-specific noise, low-light artifacts, video compression and other complex distortions.
- Synthetic degradation should be validated against real surveillance data before training a production model.

## 8. Output Directory

Generated images are stored locally in:

`outputs/images/experiment_03/`

The generated images are excluded from ordinary Git commits through the project's ignore rules.

## 9. Conclusion

This experiment successfully implements a controlled image degradation pipeline using OpenCV and NumPy.

The pipeline generates synthetic low-resolution images by combining blur, noise, JPEG compression and downsampling.

It provides a starting point for constructing HR-LR image pairs for future CNN-based and GAN-based super-resolution models.

## 10. Technologies Used

- Python
- OpenCV
- NumPy
- Matplotlib