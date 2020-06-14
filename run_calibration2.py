import pickle
import numpy
import chessboard
import park_martin
import yaml
numpy.set_printoptions(linewidth=300, suppress=True)
from scipy.linalg import expm, inv
from numpy import dot, eye
import numpy as np
import sys
import cv2 
from pathlib import Path
from tqdm import tqdm 

folder = '/Users/daniele/work/workspace_python/urx_test/output'

images = sorted(list(Path(folder).glob('*.jpg'))) 
poses = sorted(list(Path(folder).glob('*.txt'))) 

jumps=100
images = images[::jumps]
poses = poses[::jumps]


img_list = []
for image in images:
    img_list.append(cv2.imread(str(image),0))

rob_pose_list = []
for pose in poses:
    rob_pose_list.append(np.loadtxt(pose))


#if sys.version_info.major > 2:
#    img_list = pickle.load(open('data/image_list.dump', 'rb'), encoding = 'latin1')
#    rob_pose_list = pickle.load(open('data/pose_list.dump', 'rb'), encoding = 'latin1')
#else:
#    img_list = pickle.load(open('data/image_list.dump', 'rb'))
#    rob_pose_list = pickle.load(open('data/pose_list.dump', 'rb'))

corner_list = []
obj_pose_list = []
cam_pose_list = []

camera_matrix, dist_coeffs = chessboard.calibrate_lens(img_list)

def hat(v):
    return [[   0, -v[2],  v[1]],
            [v[2],     0, -v[0]],
            [-v[1],  v[0],    0]]

def tf_mat(r, t):
    res = eye(4)
    res[0:3, 0:3] = expm(hat(r))
    res[0:3, -1] = t
    return res

for i, img in tqdm(enumerate(img_list)):
    #cv2.imshow("image", img)
    #cv2.waitKey(0)
    found, corners = chessboard.find_corners(img)
    corner_list.append(corners)
    if not found:
        raise Exception("Failed to find corners in img # %d" % i)
    rvec, tvec = chessboard.get_object_pose(chessboard.pattern_points, corners, camera_matrix, dist_coeffs)
    object_pose = tf_mat(rvec, tvec)
    obj_pose_list.append(np.linalg.inv(object_pose))
    cam_pose_list.append(object_pose)

A, B = [], []
for i in tqdm(range(1,len(img_list))):
    p = rob_pose_list[i-1], obj_pose_list[i-1]
    n = rob_pose_list[i], obj_pose_list[i]
    A.append(dot(inv(p[0]), n[0]))
    B.append(dot(inv(p[1]), n[1]))


# Transformation to chessboard in robot gripper
X = eye(4)
Rx, tx = park_martin.calibrate(A, B)
X[0:3, 0:3] = Rx
X[0:3, -1] = tx

print("X: ")
print(X)

print("For validation. Printing transformations from the robot base to the camera")
print("All the transformations should be quite similar")

for i in range(len(img_list)):
    rob = rob_pose_list[i]
    obj = obj_pose_list[i]
    tmp = dot(rob, dot(X, inv(obj)))
    print(tmp)

# Here I just pick one, but maybe some average can be used instead
rob = rob_pose_list[0]
obj = obj_pose_list[0]
cam_pose = dot(dot(rob, X), inv(obj))

cam = {'rotation' : cam_pose[0:3, 0:3].tolist(),
       'translation' : cam_pose[0:3, -1].tolist(),
       'camera_matrix' : camera_matrix.tolist(),
       'dist_coeffs' : dist_coeffs.tolist()}

fp = open('camera.yaml', 'w')
fp.write(yaml.dump(cam))
fp.close()
