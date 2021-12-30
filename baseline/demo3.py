import os
import re
import math
import copy
import random
import open3d as o3d
from util import *
from util_3d import *
from config import Config

# def read_ply(filename):
#     """ read XYZRGBAL point cloud from filename PLY file """
#     plydata = PlyData.read(filename)
#     pc = plydata['vertex'].data
#     pc_array = np.array([[x, y, z, r, g, b, a, l] for x, y, z, r, g, b, a, l in pc])
#     return pc_array, plydata['face'].data

def read_ply(filename):
    """ read XYZRGBAL point cloud from filename PLY file """
    plydata = PlyData.read(filename)
    pc = plydata['vertex'].data
    pc_array = np.array([[x, y, z, r, g, b, a, l] for x, y, z, r, g, b, a, l in pc])
    return pc_array, plydata['face'].data, max(plydata.elements[0].data['label']), np.asarray(plydata.elements[0].data['label'])


def read_groundtruth(filename):
    with open(filename, 'r') as f1:
        lines = f1.readline()
        tmp = re.findall(r'\b\d+\b', lines)
    return int(tmp[0])

def intersect(x1, x2, ax1, ax2):
    if x1 > ax1:
        t = x1
        x1 = ax1
        ax1 = t
        t = x2
        x2 = ax2
        ax2 = t
    return x2 > ax1

def visual(arm_vec, arm_pos, id, ids, file_name):
    origin = o3d.io.read_point_cloud(file_name)
    n_pts = np.asarray(origin.points)
    for i in range(n_pts.shape[0]):
        n_pts[i] = vertices[i][:3]
    origin.points = o3d.utility.Vector3dVector(n_pts)
    selected_object = copy.deepcopy(origin)
    selected_pts = [vertices[ver][:3] for seg in object_id_to_segs[id] for ver in seg_to_verts[seg]]
    selected_colors = [vertices[ver][3:6] for seg in object_id_to_segs[id] for ver in seg_to_verts[seg]]
    selected_object.points = o3d.utility.Vector3dVector(selected_pts)
    selected_object.colors = o3d.utility.Vector3dVector(selected_colors)
    bbox = selected_object.get_axis_aligned_bounding_box()
    bbox.color = (0, 1, 0)
    pcds = [origin, bbox]
    for idd in ids:
        selected_object = copy.deepcopy(origin)
        selected_pts = [vertices[ver][:3] for seg in object_id_to_segs[idd] for ver in seg_to_verts[seg]]
        selected_colors = [vertices[ver][3:6] for seg in object_id_to_segs[idd] for ver in seg_to_verts[seg]]
        selected_object.points = o3d.utility.Vector3dVector(selected_pts)
        selected_object.colors = o3d.utility.Vector3dVector(selected_colors)
        bbox = selected_object.get_axis_aligned_bounding_box()
        bbox.color = (1, 0, 0)
        pcds.append(bbox)
    
    d_tmp=np.asarray([0,0,0]).reshape(1,3)
    dir_p=np.asarray([0,0,0]).reshape(1,3)
    dir_c=np.asarray([1,0,0]).reshape(1,3)
    for i in range(0,100):
        d_tmp=d_tmp+arm_vec*0.1
        dir_p=np.concatenate((dir_p, d_tmp), axis=0)
        dir_c=np.concatenate((dir_c, np.asarray([1,0,0]).reshape(1,3)), axis=0)
    direction = copy.deepcopy(origin)
    direction.points = o3d.utility.Vector3dVector(dir_p)
    direction.colors = o3d.utility.Vector3dVector(dir_c)
    direction=direction.translate(arm_pos.flatten())
    pcds.append(direction)
    
    o3d.visualization.draw_geometries(pcds)

def split(num_labels, labels, origin, num_ver1):
    pcds = []
    pcd_p = []
    pcd_c = []
    pts = np.asarray(origin.points)
    pts_c = np.asarray(origin.colors)
    for i in range(0, num_labels + 1):
        pcd_p.append(np.zeros([0,3]))
        pcd_c.append(np.zeros([0,3]))
    for i in range(num_ver1):
        pcd_p[labels[i]] = np.concatenate((pcd_p[labels[i]], pts[i:i+1, :]), axis=0)
        pcd_c[labels[i]] = np.concatenate((pcd_c[labels[i]], pts_c[i:i+1, :]), axis=0)
    return pcds, pcd_p, pcd_c

def estimateGaussian(X):
    m,n = X.shape
    mu = np.zeros((n, 1))
    sigma2 = np.zeros((n, 1))
    mu = np.mean(X, axis = 0)
    sigma2 = np.var(X, axis = 0)
    return mu, sigma2

def multivariateGaussian(X, mu, Sigma2):
    k = len(mu)
    if Sigma2.shape[0] > 1:
        Sigma2 = np.diag(Sigma2)
    X = X - mu
    argu = (2 * np.pi)**(-k / 2) * np.linalg.det(Sigma2)**(-0.5)
    p = argu * np.exp(-0.5 * np.sum(np.dot(X, np.linalg.inv(Sigma2)) * X, axis = 1))
    return p

def rdm_rotate(dir0, ang):
    ang = ang * 3.1415926535 / 180.0
    x, y, z = dir0[0], dir0[1], dir0[2]
    r = math.sqrt(x * x + y * y)
    cosang = math.cos(ang)
    sinang = math.sin(ang)
    phi = math.atan(z / r)
    # print(dir0)
    dir1 = [x * cosang - y * sinang, x * sinang + y * cosang, z]
    dir2 = [x * cosang + y * sinang, -x * sinang + y * cosang, z]
    dir3 = [x, y, math.tan(phi + ang) * r]
    dir4 = [x, y, math.tan(phi - ang) * r]
    return [dir0, dir1, dir2, dir3, dir4]
    
def dot_cross(x1, y1, z1, x2, y2, z2):
    return x1 * x2 + y1 * y2 + z1 * z2

def trans(x1, y1, z1, x2, y2, z2):
    if x1 == x2:
        return 1., 0., 0.
    elif y1 == y2:
        return 0., 1., 0.
    elif z1 == z2:
        return 0., 0., 1.

def cross_ray_plain(px, py, pz, x2, y2, z2, dir, pos):
    # print(px, py, pz)
    # print(x2, y2, z2)
    nx, ny, nz = trans(px, py, pz, x2, y2, z2)
    # print(nx, ny, nz)
    if dot_cross(nx,ny,nz,dir[0],dir[1],dir[2]) == 0.0:
        return False, None
    qx, qy, qz = 0., 0., 0.
    # print('norm ', nx,ny,nz)
    # print('p0 ', px,py,pz)
    # print('dir ', dir[0],dir[1],dir[2])
    # print(pos)
    # print('pos ', pos[0],pos[1],pos[2])
    # print(dot_cross(nx,ny,nz,px,py,pz),dot_cross(nx,ny,nz,pos[0],pos[1],pos[2]),dot_cross(nx,ny,nz,dir[0],dir[1],dir[2]))
    t = (dot_cross(nx,ny,nz,px,py,pz) - dot_cross(nx,ny,nz,pos[0],pos[1],pos[2])) / dot_cross(nx,ny,nz,dir[0],dir[1],dir[2])
    # print(t)
    qx = pos[0] + t * dir[0]
    qy = pos[1] + t * dir[1]
    qz = pos[2] + t * dir[2]
    # print(px,qx,x2)
    # print(py,qy,y2)
    # print(pz,qz,z2)
    # print()
    if px <= qx <= x2 and py <= qy <= y2 and pz <= qz <= z2:
        return True, t
    else:
        return False, None

def judge_intersect(bbox, dirr, pos):
    ret1, ret2 = False, None
    x1,y1,z1 = bbox[0], bbox[2], bbox[4]
    x2,y2,z2 = bbox[1], bbox[3], bbox[5]
    wow1, wow2 = cross_ray_plain(x1,y1,z1,x2,y2,z1, dirr, pos)
    if wow1 == True:
        if ret1 == False:
            ret1, ret2 = wow1, wow2
        else:
            ret2 = min(ret2, wow2)
    
    wow1, wow2 = cross_ray_plain(x1,y1,z2,x2,y2,z2, dirr, pos)
    if wow1 == True:
        if ret1 == False:
            ret1, ret2 = wow1, wow2
        else:
            ret2 = min(ret2, wow2)
    wow1, wow2 = cross_ray_plain(x1,y1,z1,x2,y1,z2, dirr, pos)
    if wow1 == True:
        if ret1 == False:
            ret1, ret2 = wow1, wow2
        else:
            ret2 = min(ret2, wow2)
    
    wow1, wow2 = cross_ray_plain(x1,y2,z1,x2,y2,z2, dirr, pos)
    if wow1 == True:
        if ret1 == False:
            ret1, ret2 = wow1, wow2
        else:
            ret2 = min(ret2, wow2)
    wow1, wow2 = cross_ray_plain(x1,y1,z1,x1,y2,z2, dirr, pos)
    if wow1 == True:
        if ret1 == False:
            ret1, ret2 = wow1, wow2
        else:
            ret2 = min(ret2, wow2)
    
    wow1, wow2 = cross_ray_plain(x2,y1,z1,x2,y2,z2, dirr, pos)
    if wow1 == True:
        if ret1 == False:
            ret1, ret2 = wow1, wow2
        else:
            ret2 = min(ret2, wow2)

    return ret1, ret2
    
def cal_point_line(ver, pos, dir):
    px,py,pz = ver[0],ver[1],ver[2]
    ax,ay,az = pos[0],pos[1],pos[2]
    bx,by,bz = ax+dir[0],ay+dir[1],az+dir[2]
    t = -((ax-px)*(bx-ax)+(ay-py)*(by-ay)+(az-pz)*(bz-az))/((bx-ax)**2+(by-ay)**2+(bz-az)**2)*1.0
    x= ax + (bx - ax) * t
    y= ay + (by - ay) * t
    z= az + (bz - az) * t
    ver_dis = math.sqrt((px - x) * (px - x) + (py - y) * (py - y) + (pz - z) * (pz - z))
    dis = (px - ax) * (bx - ax) + (py - ay) * (by - ay) + (pz - az) * (bz - az)
    return ver_dis, dis

if __name__ == "__main__":
    cfg = Config()
    if cfg.seed:
        np.random.seed(cfg.seed)
    
    scene_list = []
    files = os.listdir(cfg.dataset_path)
    for file in files:
        dr = os.path.join(cfg.dataset_path, file)
        if (os.path.isdir(dr)):
            scene_list.append(file)
    label_mapping = read_label_mapping(cfg.labels_dir)
    # for every scene
    suc = 0
    tot = 0
    for _ in scene_list:
        tot += 1
        prefix = _[:12]
        t_path = os.path.join(cfg.dataset_path, _)

        # read files
        t_dir = os.path.join(t_path, prefix + cfg.ply_suffix)
        ply = t_dir
        vertices, faces, n_labels, labels = read_ply(t_dir)
        t_dir = os.path.join(t_path, cfg.ans_name)
        ans = read_groundtruth(t_dir)
        t_dir = os.path.join(t_path, prefix + cfg.agg_suffix)
        object_id_to_segs, label_to_segs = read_aggregation(t_dir)
        t_dir = os.path.join(t_path, prefix + cfg.seg_suffix)
        seg_to_verts, num_verts = read_segmentation(t_dir)

        # point cloud of human
        origin = o3d.io.read_point_cloud(ply)
        pcds, pcd_p, pcd_c = split(n_labels, labels, origin, num_verts)
        pcd2 = copy.deepcopy(origin)
        nnn = len(vertices)
        selected_pts = [vertices[ver][:3] for ver in range(nnn) if labels[ver] == 42]
        selected_colors = [vertices[ver][3:6] for ver in range(nnn) if labels[ver] == 42]
        pcd2.points = o3d.utility.Vector3dVector(selected_pts)
        pcd2.colors = o3d.utility.Vector3dVector(selected_colors)

        # anomaly detection
        arm_pos = np.asarray([0,0,0])
        arm_end = np.asarray([0,0,0])  
        arm_p = np.zeros([0,3]) 

        pcd2_down = pcd2.voxel_down_sample(0.02)
        pcd2_down.paint_uniform_color([0.7, 0.7, 0.7])
        test = copy.deepcopy(pcd2_down).translate((0,1,0))

        pcd2_down_p = np.asarray(pcd2_down.points)
        pts = np.asarray(test.points)
        pts_c = np.asarray(test.colors)
        pts[0:,2:] = 0

        X = pts[0:,0:2]
        m, n = X.shape
        mu, sigma2 = estimateGaussian(X)

        p = multivariateGaussian(X, mu, sigma2)
        eps = 0.8
        eps2 = 0.2
        cnt = 0
        cnt2 = 0
        for i in range(0, m):
            if p[i] < eps:
                #pts_c[i:i+1,0:]=[1,0,0]
                arm_pos = arm_pos + np.asarray(pcd2_down_p[i:i+1,0:])
                arm_p = np.concatenate((arm_p, pcd2_down_p[i:i+1,0:]), axis=0)
                cnt = cnt + 1
            if p[i] < eps2:
                #pts_c[i:i+1,0:]=[1,0,1]
                arm_end = arm_end + np.asarray(pcd2_down_p[i:i+1,0:])
                cnt2 = cnt2 + 1
        if cnt > 0:
            arm_pos = arm_pos / cnt
        if cnt2 > 0:
            arm_end = arm_end / cnt2

        test.points = o3d.utility.Vector3dVector(pts)
        test.colors = o3d.utility.Vector3dVector(pts_c)
        pcd2_down.colors = o3d.utility.Vector3dVector(pts_c)

        arm_vec = (arm_end - arm_pos).flatten()
        # print(arm_end, arm_pos, arm_vec)
        div = math.sqrt(arm_vec[0] ** 2 + arm_vec[1] ** 2 + arm_vec[2] ** 2)
        arm_vec = arm_vec / div
        arm_vec = arm_vec.reshape(3)

        # traverse all points
        object_id = [0 for _ in range(len(vertices))]
        n_points_per_id = [0 for _ in range(len(object_id_to_segs) + 1)]
        for j in range(1, len(object_id_to_segs) + 1):
            for seg in object_id_to_segs[j]:
                n_points_per_id[j] += len(seg_to_verts[seg])
                for ver in seg_to_verts[seg]:
                    object_id[ver] = j

        pos = arm_pos.reshape(3)
        objs = []
        Len = 0.
        for i, p in enumerate(vertices):
            if object_id[i] == 0:
                continue
            dis, leng = cal_point_line(p, pos, arm_vec)
            Len = max(Len, leng)
        for i, p in enumerate(vertices):
            if object_id[i] == 0:
                continue
            dis, leng = cal_point_line(p, pos, arm_vec)
            if dis < 0.4 and leng > 0:  # parameter
                objs.append(object_id[i])
        if objs == []:
            objs = [1]
        #visual(arm_vec, arm_pos, ans, [], ply)

        # majority voting
        objs_uni = np.unique(objs, return_index=False)
        counts = np.bincount(objs)
        #dens = [counts[i] / n_points_per_id[i] for i in range(len(counts))]
        #result = np.argmax(dens[1:]) + 1
        result = np.argmax(counts)
        if result == ans:
            suc += 1
        
        print('for prefix={}:'.format(prefix))
        # print('dens={}'.format(dens))
        print('counts={}'.format(counts))
        print('ans={}, result={}'.format(ans, result))
        print('suc/tot={}/{}={}'.format(suc, tot, suc/tot))
        print()
        # print('objs={}'.format(objs))
        # print('counts={}'.format(counts))
        # print()

        # visualization
        # print(objs_uni)
        # visual(arm_vec, arm_pos, ans, [result], ply)

        #break
    print('suc/tot={}/{}={}'.format(suc, tot, suc/tot))

        