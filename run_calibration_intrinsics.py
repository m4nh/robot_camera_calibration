from tqdm import tqdm
import cv2
from pathlib import Path
import sys
import numpy as np
from numpy import dot, eye
from scipy.linalg import expm, inv
import pickle
import numpy
import chessboard
import park_martin
import yaml
numpy.set_printoptions(linewidth=300, suppress=True)

folder = '/private/tmp/outboard/'

images = sorted(list(Path(folder).glob('*.jpg')))

img_list = []
for image in images:
    img_list.append(cv2.imread(str(image), 0))


camera_matrix, dist_coeffs = chessboard.calibrate_lens(img_list)

data = {
    'intrinsics': {
        'camera_matrix': camera_matrix.tolist(),
        'dist_coeffs': dist_coeffs.tolist()
    }
}

print(camera_matrix)
print(dist_coeffs)

yaml.safe_dump(data, open('/tmp/camera.yml', 'w'))
