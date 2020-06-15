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


def tf_mat(r, t):
    res = eye(4)
    res[0:3, 0:3] = expm(hat(r))
    res[0:3, -1] = t
    return res


def mat_2_cv(m):
    rot = m[:3, :3]
    rvec, _ = cv2.Rodrigues(rot)
    tvec = m[:3, 3]
    return rvec, tvec


# Configuration file
calibration_file = 'camera.yaml'
calibration_cfg = yaml.safe_load(open(calibration_file, 'r'))
print(calibration_cfg)

# Camera Intrinsics
camera_matrix = np.array(calibration_cfg['intrinsics']['camera_matrix'])
dist_coeffs = np.array(calibration_cfg['intrinsics']['dist_coeffs'])
A = np.eye(4)
A[:3, :3] = camera_matrix

# Target Pose
target_pose = np.eye(4)
target_pose[:3, :3] = np.array(calibration_cfg['target_pose']['rotation'])
target_pose[:3, -1] = np.array(calibration_cfg['target_pose']['translation'])


relative_t2 = np.eye(4)
relative_t2[0, 3] = 8 * 0.0245


other_target_pose = np.dot(target_pose, relative_t2)

targets = [target_pose, other_target_pose]


# Camera Pose
ee_to_camera = np.eye(4)
ee_to_camera[:3, :3] = np.array(calibration_cfg['camera_pose']['rotation'])
ee_to_camera[:3, -1] = np.array(calibration_cfg['camera_pose']['translation'])

print(target_pose)
print(ee_to_camera)


# Load Dataset
folder = '/Users/daniele/work/workspace_python/urx_test/output'
images = sorted(list(Path(folder).glob('*.jpg')))
poses = sorted(list(Path(folder).glob('*.txt')))
jumps = 50
images = images[::jumps]
poses = poses[::jumps]


rob_pose_list = []
for pose in poses:
    rob_pose_list.append(np.loadtxt(pose))


corner_list = []
obj_pose_list = []
cam_pose_list = []

for i, image in enumerate(images):
    img = cv2.imread(str(image))
    img = cv2.undistort(img, A[:3, :3], dist_coeffs)

    camera_pose = np.dot(rob_pose_list[i], ee_to_camera)

    for target in targets:
        camera_to_target = np.dot(np.linalg.inv(camera_pose), target)
        print(camera_to_target)

        uv = np.dot(A, camera_to_target[:4, 3])
        uv = uv[:3]
        uv = uv / uv[2]
        uv = uv.astype(np.int)
        print(uv)
        cv2.circle(img, (uv[0], uv[1]), 1, (255, 0, 200), thickness=3)

        rvec, tvec = mat_2_cv(camera_to_target)
        obj_points = np.array([0., 0., 0.]).reshape((3, -1))

        print("OBJE POINTS", obj_points)
        image_points, _ = cv2.projectPoints(obj_points, rvec, tvec, A[:3, :3], dist_coeffs)
        print("image_points", image_points)
        image_points = image_points.ravel()
       # print("image_points", image_points)
        image_points = image_points.astype(int)
        cv2.circle(img, (image_points[0], image_points[1]), 1, (0, 255, 200), thickness=3)
        # for ip in image_points:
        #    ip = ip.astype(int)
        #    print(ip)
        #cv2.circle(img, (ip[0], ip[1]), 1, (0, 255, 200), thickness=3)

        print("IMAGE POINTS", image_points.shape)

    """
    AP = np.dot(A, camera_pose)

    
    

    cv2.circle(img, (uv[1], uv[0]), 10, (255, 0, 200))

    print(camera_pose)
    print(AP)
    print(point)
    print(uv)
    """

    cv2.imshow("image", img)
    c = cv2.waitKey(0)
    if c == ord('q'):
        break
