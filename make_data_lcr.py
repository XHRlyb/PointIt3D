import argparse
import math
import os
import numpy as np
import open3d as o3d
import copy
from plyfile import PlyData, PlyElement
from util import *
from util_3d import *


def write_ply(points, face, filename, text=True):
	""" input: Nx3, Nx5(rgbal) write points and colors to filename as PLY format. """
	num_points = len(points)

	points = [(points[i, 0], points[i, 1], points[i, 2], points[i, 3], points[i, 4], points[i, 5], points[i, 6], points[i, 7]) for i in range(points.shape[0])]
	vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), ('alpha', 'u1'), ('label', 'u2')])

	vertex_all = np.empty(num_points, vertex.dtype.descr)

	for prop in vertex.dtype.names:
		vertex_all[prop] = vertex[prop]

	el = [PlyElement.describe(vertex_all, 'vertex'), PlyElement.describe(face, 'face')]

	PlyData(el, text=text).write(filename)


def read_ply(filename):
	""" read XYZRGBAL point cloud from filename PLY file """
	plydata = PlyData.read(filename)
	pc = plydata['vertex'].data
	pc_array = np.array([[x, y, z, r, g, b, a, l] for x, y, z, r, g, b, a, l in pc])
	return pc_array, plydata['face'].data

def rotate_a_eric(obj_x, obj_y, obj_z, foot_x = 0.0, foot_y = 0.0):
	arm_vec_x, arm_vec_y, arm_vec_z = -0.02023186, -0.05549748, 1.51
	hand_x, hand_y, hand_z = foot_x + arm_vec_x, foot_y + arm_vec_y, arm_vec_z
	obj_dir_x, obj_dir_y = obj_x - foot_x, obj_y - foot_y
	costhe = (arm_vec_x * obj_dir_x + arm_vec_y * obj_dir_y) / (math.sqrt(arm_vec_x * arm_vec_x + arm_vec_y * arm_vec_y) * math.sqrt(obj_dir_x * obj_dir_x + obj_dir_y * obj_dir_y))
	sinthe = (arm_vec_x * obj_dir_y - arm_vec_y * obj_dir_x) / (math.sqrt(arm_vec_x * arm_vec_x + arm_vec_y * arm_vec_y) * math.sqrt(obj_dir_x * obj_dir_x + obj_dir_y * obj_dir_y))
	tanphi = (obj_z - hand_z) / math.sqrt((obj_x - foot_x) * (obj_x - foot_x) + (obj_y - foot_y) * (obj_y - foot_y))
	phi = math.atan(tanphi) * 180 / 3.1415926535
	eric_name = None
	print(costhe, sinthe)
	phi_0 = int((phi + 2.5) / 5) * 5
	print(obj_x, obj_y, obj_z, hand_z, phi, phi_0)
	if phi_0 <= 15 and phi_0 >= -30: 
		eric_name = 'eric_{}.ply'.format(-phi_0)#phi_0
	return eric_name, costhe, sinthe
	#return eric_name, 1, 0

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_dir", default='ScanNet/scans', type=str, help="path to ScanNet folder")
	parser.add_argument("--labels_dir", default='ScanNet/scannetv2-labels.combined.tsv', type=str, help="path to scannetv2-labels.combined.tsv")
	parser.add_argument("--scanID",
	                    help="path to the scene folder to be processed",
	                    required=True,
	                    type=str)
	args = parser.parse_args()
	data_dir = os.path.join(args.data_dir, args.scanID)
	output_dir = os.path.join('ScanNet_with_eric', args.scanID + '_with_eric')
	mesh_with_labels_file = os.path.join(data_dir, args.scanID + '_vh_clean_2.labels.ply')
	eric_file = 'eric.ply'
	meta_file = os.path.join(data_dir, args.scanID + ".txt")
	aggregation_file = os.path.join(data_dir, args.scanID + ".aggregation.json")
	segs_file = os.path.join(data_dir, args.scanID + "_vh_clean_2.0.010000.segs.json")

	np.random.seed(42)


	# read files
	object_id_to_segs, label_to_segs = read_aggregation(aggregation_file)  # 'floor' -> 6555 (object_id is 1-indexed)
	seg_to_verts, num_verts = read_segmentation(segs_file)  # 6555 -> 3
	label_mapping = read_label_mapping(args.labels_dir)  # 'floor' ->  2
	vertices, faces = read_ply(mesh_with_labels_file)

	# aligns the scene to origin point

	lines = open(meta_file).readlines()
	axis_align_matrix = None
	for line in lines:
		if 'axisAlignment' in line:
			axis_align_matrix = [float(x) for x in line.rstrip().strip('axisAlignment = ').split(' ')]
	if axis_align_matrix:
		axis_align_matrix = np.array(axis_align_matrix).reshape((4, 4))
		pts = np.ones((vertices.shape[0], 4))
		pts[:, :3] = vertices[:, :3]
		pts = np.dot(pts, axis_align_matrix.transpose())
		vertices[:, :3] = pts[:, :3]

	# choose the object to be pointed to and the angle the person should rotate

	id = np.random.randint(1, len(object_id_to_segs) + 1)
	# id = 28
	minx, maxx, miny, maxy, minz, maxz = 0, 0, 0, 0, 0, 0
	for seg in object_id_to_segs[id]:
		for i, ver in enumerate(seg_to_verts[seg]):
			#print(ver, vertices[ver])
			if i == 0:
				minx = vertices[ver][0]
				maxx = vertices[ver][0]
				miny = vertices[ver][1]
				maxy = vertices[ver][1]
				minz = vertices[ver][2]
				maxz = vertices[ver][2]
			else:
				minx = min(minx, vertices[ver][0])
				maxx = max(maxx, vertices[ver][0])
				miny = min(miny, vertices[ver][1])
				maxy = max(maxy, vertices[ver][1])
				minz = min(minz, vertices[ver][2])
				maxz = max(maxz, vertices[ver][2])
	cx = (minx + maxx) / 2
	cy = (miny + maxy) / 2
	cz = (minz + maxz) / 2

	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	with open(os.path.join(output_dir, 'answer.txt'), 'w') as f:
		f.write(str(id))

	# # visualization
	# origin = o3d.io.read_point_cloud(mesh_with_labels_file)
	# n_pts = np.asarray(origin.points)
	# for i in range(n_pts.shape[0]):
	# 	n_pts[i] = vertices[i][:3]
	# origin.points = o3d.utility.Vector3dVector(n_pts)
	# selected_object = copy.deepcopy(origin)
	# selected_pts = [vertices[ver][:3]for ver in seg_to_verts[seg] for seg in object_id_to_segs[id]]
	# selected_colors = [vertices[ver][4:7]for ver in seg_to_verts[seg] for seg in object_id_to_segs[id]]
	# selected_object.points = o3d.utility.Vector3dVector(selected_pts)
	# selected_object.colors = o3d.utility.Vector3dVector(selected_colors)
	# bbox = selected_object.get_axis_aligned_bounding_box()
	# bbox.color = (0, 0, 0)
	# pcds = [origin, bbox]
	# o3d.visualization.draw_geometries(pcds)

	# process person
	eric_name, costhe, sinthe = rotate_a_eric(cx, cy, cz)
	if eric_name == None:
		print('NO SUIT ERIC!')
		exit(0)
	print(eric_name, id)
	eric_file = 'eric_70/' + eric_name
	with open(eric_file, 'rb') as f:
		ply_eric = PlyData.read(f)
		num_verts1 = ply_eric['vertex'].count
		num_faces = ply_eric['face'].count
		ver = []
		fac = []
		for i in range(num_verts1):
			x, y, z, _, _, _, _, _ = ply_eric['vertex'][i]
			xx = x * costhe - y * sinthe
			yy = x * sinthe + y * costhe
			x = xx
			y = yy
			ver.append((x, y, z, 255, 0, 0, 240, 42))
		vertices = np.concatenate((vertices, np.array(ver, dtype=vertices.dtype)))

		for i in range(num_faces):
			fac.append([x + num_verts for x in ply_eric['face'][i][0]])

		new_fac = np.empty(num_faces, dtype=faces.dtype)
		new_fac['vertex_indices'] = fac
		faces = np.concatenate((faces, new_fac))
	vertices = np.concatenate((vertices, np.array([[cx, cy, cz, 255, 0, 0, 240, 111]], dtype=vertices.dtype)))
	write_ply(vertices, faces, os.path.join(output_dir, args.scanID + '_with_eric.ply'), False)

if __name__ == "__main__":
	main()

'''
# np.set_printoptions(threshold=np.inf)
'''