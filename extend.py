import os

import numpy
import open3d as o3d
import numpy as np
import copy
import random
import math
from plyfile import PlyData, PlyElement

data_dir = ''
input_file = data_dir + 'merge.ply'

# the first 2 parts seem a little time-consuming
# maybe some optimizations can be made
# e.g. to reduce the number of points in the input file by down-sampling
print('reading ply...')
with open(input_file, 'rb') as f:
	plydata = PlyData.read(f)

num_ver1 = plydata['vertex'].count
vertex = plydata['vertex'].data
face = plydata['face'].data

num_labels = max(plydata.elements[0].data['label'])

labels = np.asarray(plydata.elements[0].data['label'])

print('completed')
print('locating the person...')
# this part splits the point cloud by labels
# only the person are used in the following steps
pcds = []
pcd_p = []
pcd_c = []
origin = o3d.io.read_point_cloud(input_file)

def split():
	pts = np.asarray(origin.points)
	pts_c = np.asarray(origin.colors)
	for i in range(0, num_labels + 1):
		pcd_p.append(np.zeros([0,3]))
		pcd_c.append(np.zeros([0,3]))

	for i in range(num_ver1):
		pcd_p[labels[i]] = np.concatenate((pcd_p[labels[i]], pts[i:i+1, :]), axis=0)
		pcd_c[labels[i]] = np.concatenate((pcd_c[labels[i]], pts_c[i:i+1, :]), axis=0)


split()
pcd2 = copy.deepcopy(origin) # to store the person
print('completed')

print('doing some calculations...')
for i in range(1, num_labels + 1):
	if pcd_p[i].size > 0:
		tmp = copy.deepcopy(origin)
		tmp.points = o3d.utility.Vector3dVector(pcd_p[i])
		tmp.colors = o3d.utility.Vector3dVector(pcd_c[i])
		bbox = tmp.get_axis_aligned_bounding_box()
		bbox.color = (random.random(),random.random(),random.random())
		if i == 42: # the person
			pcds.append(bbox)
			pcd2 = tmp
		else:
			pcds.append(tmp)
	#o3d.visualization.draw_geometries([tmp])

# locating the pointing arm by an Anomaly Detection algorithm
# the parameters are set manually right now
def estimateGaussian(X):
	m,n = X.shape
	mu = np.zeros((n,1))
	sigma2 = np.zeros((n,1))

	mu = np.mean(X, axis=0)
	sigma2 = np.var(X,axis=0)
	return mu,sigma2

def multivariateGaussian(X,mu,Sigma2):
	k = len(mu)
	if (Sigma2.shape[0]>1):
		Sigma2 = np.diag(Sigma2)
	X = X-mu
	argu = (2*np.pi)**(-k/2)*np.linalg.det(Sigma2)**(-0.5)
	p = argu*np.exp(-0.5*np.sum(np.dot(X,np.linalg.inv(Sigma2))*X,axis=1))
	return p

# the pointing direction is a ray from arm_pos to arm_end
arm_pos = np.asarray([0,0,0])
arm_end = np.asarray([0,0,0])

# to store the points in the arm(not used in this algorithm)
arm_p = np.zeros([0,3])

def processing_person():
	global arm_pos
	global arm_end
	global arm_p
	pcd2_down = pcd2.voxel_down_sample(0.02)
	pcd2_down.paint_uniform_color([0.7, 0.7, 0.7])
	test = copy.deepcopy(pcd2_down).translate((0,1,0))

	pcd2_down_p = np.asarray(pcd2_down.points)
	pts = np.asarray(test.points)
	pts_c = np.asarray(test.colors)
	pts[0:,2:]=0

	X=pts[0:,0:2]
	m,n = X.shape
	mu,sigma2=estimateGaussian(X)

	p=multivariateGaussian(X,mu,sigma2)
	eps = 0.8
	eps2 = 0.2
	cnt = 0
	cnt2 = 0
	for i in range(0, m):
		if p[i]<eps:
			#pts_c[i:i+1,0:]=[1,0,0]
			arm_pos=arm_pos+np.asarray(pcd2_down_p[i:i+1,0:])
			arm_p=np.concatenate((arm_p, pcd2_down_p[i:i+1,0:]), axis=0)
			cnt=cnt+1
		if p[i]<eps2:
			#pts_c[i:i+1,0:]=[1,0,1]
			arm_end=arm_end+np.asarray(pcd2_down_p[i:i+1,0:])
			cnt2=cnt2+1
	if cnt > 0:
		arm_pos = arm_pos/cnt
	if cnt2 > 0:
		arm_end = arm_end/cnt2

	test.points = o3d.utility.Vector3dVector(pts)
	test.colors = o3d.utility.Vector3dVector(pts_c)
	pcd2_down.colors = o3d.utility.Vector3dVector(pts_c)
	pcds.append(pcd2_down)

processing_person()

arm_vec = (arm_end - arm_pos).flatten()
print(arm_end,arm_pos)

div = math.sqrt(arm_vec[0]**2+arm_vec[1]**2+arm_vec[2]**2)
arm_vec = arm_vec/div
arm_vec = arm_vec.reshape(1,3)

# visualizating the pointing direction
d_tmp=np.asarray([0,0,0]).reshape(1,3)
dir_p=np.asarray([0,0,0]).reshape(1,3)
dir_c=np.asarray([1,0,0]).reshape(1,3)
for i in range(0,30):
	d_tmp=d_tmp+arm_vec*0.1
	dir_p=np.concatenate((dir_p, d_tmp), axis=0)
	dir_c=np.concatenate((dir_c, np.asarray([1,0,0]).reshape(1,3)), axis=0)
direction = copy.deepcopy(origin)
direction.points = o3d.utility.Vector3dVector(dir_p)
direction.colors = o3d.utility.Vector3dVector(dir_c)
direction=direction.translate(arm_pos.flatten())
pcds.append(direction)

print('completed')
o3d.visualization.draw_geometries(pcds) 
