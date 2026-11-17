
import sys
import torch
import cv2
import numpy as np
import pandas as pd
import matplotlib

print("=" * 50)
print("SURVEILLANCE SUPER-RESOLUTION")
print("ENVIRONMENT VERIFICATION")
print("=" * 50)

print("\nPython:", sys.version)
print("Python executable:", sys.executable)

print("\nPyTorch:", torch.__version__)
print("OpenCV:", cv2.__version__)
print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
print("Matplotlib:", matplotlib.__version__)

print("\nHardware Information")

print("CUDA available:", torch.cuda.is_available())

if hasattr(torch, "xpu"):
    print("Intel XPU available:", torch.xpu.is_available())

print("\nTensor Test")

x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)

print(x)
print("Tensor shape:", x.shape)
print("Tensor device:", x.device)

print("\nEnvironment verification completed!")
