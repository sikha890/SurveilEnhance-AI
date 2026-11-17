# Experiment 04: HR-LR Dataset Generation and Validation

## 1. Objective

The objective of this experiment is to build a reusable pipeline that generates paired High-Resolution (HR) and Low-Resolution (LR) images for training and evaluating image super-resolution models.

Each HR image acts as the reference image, while its corresponding LR image is generated through a controlled degradation process.

## 2. Dataset Structure

```text
data/
├── source/
│   └── original_hr.png
└── generated/
    ├── HR/
    │   └── original_hr.png
    └── LR/
        └── original_hr.png
```

The source image is retained separately. Generated HR and LR images use matching filenames to preserve their pairing.

## 3. Degradation Pipeline

The LR image is generated through the following sequence:

1. Gaussian blur
2. Gaussian noise
3. JPEG compression
4. Downsampling using OpenCV INTER_AREA interpolation

The configuration is maintained in `configs/degradation.yaml`.

| Parameter | Value |
|---|---:|
| Scale factor | 4× |
| Gaussian blur kernel | 5 × 5 |
| Gaussian blur sigma | 1.0 |
| Gaussian noise standard deviation | 10 |
| JPEG quality | 40 |
| Random seed | 42 |

## 4. HR-LR Dimensions

The source image has dimensions of 256 × 256 pixels.

After downsampling by a factor of four:

- HR dimensions: 256 × 256
- LR dimensions: 64 × 64

The relationship is:

LR width = HR width / scale factor

LR height = HR height / scale factor

## 5. Implementation Files

| File | Responsibility |
|---|---|
| `configs/degradation.yaml` | Stores degradation parameters |
| `src/data/degradation.py` | Implements reusable degradation functions |
| `src/data/dataset_generator.py` | Generates paired HR-LR images |
| `src/data/validate_dataset.py` | Checks pairing and image dimensions |
| `tests/test_degradation.py` | Tests degradation functions |

## 6. Validation Results

The dataset validation script completed successfully.

| Validation metric | Result |
|---|---:|
| Total valid pairs | 1 |
| Validation errors | 0 |
| HR dimensions | 256 × 256 |
| LR dimensions | 64 × 64 |
| Status | Passed |

## 7. Automated Testing

The automated test suite contains eight tests:

1. Gaussian blur preserves image shape and data type.
2. Gaussian noise is reproducible with the same seed.
3. Zero noise preserves the original image.
4. JPEG compression preserves image shape and data type.
5. Downsampling produces the expected dimensions.
6. Downsampling rejects invalid image dimensions.
7. Downsampling rejects an invalid scale factor.
8. The complete degradation pipeline produces the expected output.

Test execution:

```powershell
python -m unittest tests.test_degradation -v
```

Result: 8 tests passed, 0 failures.

## 8. Limitations

- The current dataset contains only one synthetic source image.
- Synthetic degradation does not represent every real CCTV camera condition.
- Additional images and diverse degradation settings are required for meaningful model training.
- The generated images are experimental data, not evidence of authentic recovered visual details.

## 9. Conclusion

A reusable HR-LR dataset generation pipeline has been implemented and tested successfully. The pipeline generates paired images, validates their dimensions and pairing, and provides automated tests for its core degradation functions.

This experiment establishes the data preparation foundation for future super-resolution model development.


# Experiment 06 — DIV2K Dataset Preparation

## 1. Objective

Prepare a reproducible high-resolution and low-resolution
paired dataset using DIV2K for training and evaluating
the SRCNN super-resolution model.

## 2. Dataset

Source: DIV2K

| Split | Original HR Images |
|---|---:|
| Training | 800 |
| Validation | 100 |
| Total | 900 |

Original images are retained separately from generated pairs.

## 3. Degradation Pipeline

Each HR image is processed using:

1. Gaussian blur (5x5 kernel, sigma 1.0)
2. Gaussian noise (standard deviation 10)
3. JPEG compression (quality 40)
4. 4x downsampling using INTER_AREA

Images are cropped at the bottom and right edges when
necessary to make their dimensions divisible by four.

## 4. Output Structure

data/DIV2K/
    train/
        HR/
        LR/
    validation/
        HR/
        LR/

## 5. Generation Results

| Metric | Result |
|---|---:|
| Training pairs generated | 800 |
| Validation pairs generated | 100 |
| Generation failures | 0 |
| Total pairs | 900 |

## 6. Validation Results

The dataset validator checks:

- Matching HR and LR filenames
- Image readability
- Three-channel color images
- uint8 image data
- HR dimensions divisible by four
- Exact 4x HR-to-LR dimensions

Validation errors: 0

## 7. Automated Testing

The complete test suite passed:

| Test Module | Tests Passed |
|---|---:|
| Image degradation | 8 |
| DIV2K pipeline | 6 |
| SRCNN model | 6 |
| SRCNN dataset | 5 |
| Total | 25 |

## 8. Reproducibility

The degradation pipeline uses a configured random seed.
Training and validation use separate seed ranges.

The downloaded archives, extracted images, and generated
datasets are excluded from Git.

## 9. Limitations

DIV2K contains natural images and does not represent all
surveillance camera degradation conditions.

CCTV-specific validation is necessary before making
claims about real surveillance footage enhancement.
