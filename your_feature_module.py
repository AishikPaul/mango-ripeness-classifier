
import cv2
import os
import cv2
import numpy as np
import pandas as pd
from scipy.stats import skew
from skimage.feature import local_binary_pattern
import mahotas
import cv2
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

import os
import cv2
import numpy as np
import pandas as pd
from scipy.stats import skew
from skimage.feature import local_binary_pattern
# from tqdm import tqdm
import mahotas
def extract_features(image):
    features = []
    image = cv2.resize(image, (100, 100))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab  = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # RGB mean, std, skewness
    for i in range(3):
        ch = image[:, :, i].flatten()
        features.extend([np.mean(ch), np.std(ch), skew(ch)])

    # HSV mean + std
    for i in range(3):
        ch = hsv[:, :, i].flatten()
        features.extend([np.mean(ch), np.std(ch)])

    # LAB mean + std
    for i in range(3):
        ch = lab[:, :, i].flatten()
        features.extend([np.mean(ch), np.std(ch)])

    # Gray mean + std
    features.extend([np.mean(gray), np.std(gray)])

    # Yellow percentage
    h, s, v = cv2.split(hsv)
    yellow_pct = np.sum((h > 20) & (h < 40)) / h.size
    features.append(yellow_pct)

    # Dark pixel percentage
    dark_pct = np.sum(v < 50) / v.size
    features.append(dark_pct)

    # LBP texture features
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    features.extend(hist)

    # Edge density
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / edges.size
    features.append(edge_density)

    # ===== STEP 1: Color Histograms =====
    for i in range(3):  # R, G, B
        hist = cv2.calcHist([image], [i], None, [16], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        features.extend(hist)

    # ===== STEP 2: Shape Features =====
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter != 0 else 0
        features.extend([area, perimeter, circularity])
    else:
        features.extend([0, 0, 0])

    # ===== STEP 3: Haralick Texture =====
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    features.extend(haralick)

    return features