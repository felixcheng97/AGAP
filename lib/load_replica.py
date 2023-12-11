import os
import torch
import numpy as np
import imageio
import torch.nn.functional as F

def get_pose(idx_list, ref_idx, baseline=0.1):
    pose_list = []
    h_ref, w_ref = ref_idx//9, ref_idx%9
    
    r_mat = torch.FloatTensor([
        [0.,  0., -1.],
        [1.,  0.,  0.],
        [0., -1.,  0.],
    ])
    for idx in idx_list:
        h_idx, w_idx = idx//9, idx % 9
        pose = torch.FloatTensor(
            [0, baseline*(w_ref - w_idx), -baseline*(h_ref - h_idx)]).view(3, 1)
        pose = torch.cat([r_mat, pose], dim=1)
        pose_list.append(pose)
    pose_list = torch.stack(pose_list)
    return pose_list

def load_replica_data(basedir, movie_render_kwargs):
    files = []
    for h in range(0, 9):
        for w in range(0, 9):
            fname = os.path.join(basedir, f'image_{h}_{w}.png')
            files.append(fname)
            
    train_idx = [0, 2, 6, 8, 18, 26, 40, 54, 62, 72, 74, 80]
    val_idx = [1, 5, 7, 37, 41, 43, 71, 77, 79]
    test_idx = [4, 20, 22, 24, 36, 38, 42, 44, 56, 58, 60, 76]
    all_idx = train_idx + val_idx + test_idx
    all_poses = get_pose(all_idx, 40).numpy()
    
    i_train = list(range(0, len(train_idx)))
    i_val = list(range(len(train_idx), len(train_idx) + len(val_idx)))
    i_test = list(range(len(train_idx) + len(val_idx), len(train_idx) + len(val_idx) + len(test_idx)))
    i_split = [i_train, i_val, i_test]

    file_train = [files[i] for i in train_idx]
    file_val = [files[i] for i in val_idx]
    file_test = [files[i] for i in test_idx]
    file_all = file_train + file_val + file_test

    all_imgs = [(imageio.imread(f) / 255).astype(np.float32) for f in file_all]

    imgs = np.stack(all_imgs, 0)
    poses = all_poses

    H = movie_render_kwargs['H']
    W = movie_render_kwargs['W']
    wFOV = movie_render_kwargs['FOV']
    focal = 0.5*W/np.tan(0.5*np.radians(wFOV))

    K = None

    ## Get spiral
    # Get average pose
    up = np.array([0., 0., -1.])

    # define a reasonable "focal depth"
    focal_depth = 2.0

    # Get radii for spiral path
    zdelta = movie_render_kwargs.get('zdelta', 0.5)
    zrate = movie_render_kwargs.get('zrate', 1.0)
    rads = np.array([0.2, 0.2, 0.])
    # c2w_path = np.array([
    #     [0.,  0., -1., 0., H],
    #     [1.,  0.,  0., 0., W],
    #     [0., -1.,  0., 0., focal],
    # ])
    c2w_path = get_pose([41], 40).numpy()[0]
    N_rots = movie_render_kwargs.get('N_rots', 1)
    
    N_pyr = 240
    render_poses = render_path_pitch_yaw_roll(c2w_path, beta_start=0, beta_end=2*np.pi, N=N_pyr)

    # N_spiral = 120
    # N_pyr = 31

    # render_poses = []
    # render_poses += render_path_pitch_yaw_roll(c2w_path, beta_start=0, beta_end=np.pi/2, N=N_pyr)
    # render_poses += render_path_spiral(render_poses[-1], up, rads, focal_depth, zdelta, zrate=zrate, rots=N_rots, N=N_spiral)

    # render_poses += render_path_pitch_yaw_roll(c2w_path, beta_start=np.pi/2, beta_end=np.pi, N=N_pyr)
    # render_poses += render_path_spiral(render_poses[-1], up, rads, focal_depth, zdelta, zrate=zrate, rots=N_rots, N=N_spiral)
    
    # render_poses += render_path_pitch_yaw_roll(c2w_path, beta_start=np.pi, beta_end=3*np.pi/2, N=N_pyr)
    # render_poses += render_path_spiral(render_poses[-1], up, rads, focal_depth, zdelta, zrate=zrate, rots=N_rots, N=N_spiral)

    # render_poses += render_path_pitch_yaw_roll(c2w_path, beta_start=3*np.pi/2, beta_end=2*np.pi, N=N_pyr)
    # render_poses += render_path_spiral(render_poses[-1], up, rads, focal_depth, zdelta, zrate=zrate, rots=N_rots, N=N_spiral)

    # render_poses += render_path_pitch_yaw_roll(c2w_path, alpha_start=0, alpha_end=2*np.pi, N=N_pyr)

    render_poses = torch.Tensor(render_poses)

    return imgs, poses, [H, W, focal], render_poses, i_split, K


def normalize(x):
    return x / np.linalg.norm(x)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def render_path_pitch_yaw_roll(c2w, alpha_start=0, alpha_end=0, beta_start=0, beta_end=0, gamma_start=0, gamma_end=0, N=120):
    render_poses = []
    for alpha, beta, gamma in zip(np.linspace(alpha_start, alpha_end, N), np.linspace(beta_start, beta_end, N), np.linspace(gamma_start, gamma_start, N)):
        R = get_rotation_pitch_yaw_roll(alpha, beta, gamma)
        render_poses.append(np.concatenate([np.dot(c2w[:, :3], R), c2w[:, 3:]], axis=-1))

    return render_poses

# u (pitch, +y axis), v (yaw, +z axis), w (roll, +x axis)
def get_rotation_pitch_yaw_roll(alpha=0, beta=0, gamma=0):
    R_pitch = np.array([
        [1.,            0.,             0.],
        [0., np.cos(alpha), -np.sin(alpha)],
        [0., np.sin(alpha),  np.cos(alpha)]
    ])

    R_yaw = np.array([
        [np.cos(beta), 0., -np.sin(beta)],
        [          0., 1.,             0],
        [np.sin(beta), 0.,  np.cos(beta)]
    ])

    R_roll = np.array([
        [np.cos(gamma), -np.sin(gamma), 0.],
        [np.sin(gamma),  np.cos(gamma), 0.],
        [           0.,             0., 1.],
    ])

    R = np.dot(R_roll, np.dot(R_yaw, R_pitch))
    return R

def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    render_poses = []
    rads = np.array(list(rads) + [1.])
    hwf = c2w[:,4:5]

    for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]:
        c = np.dot(c2w[:3,:4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta*zrate)*zdelta, 1.]) * rads) 
        z = normalize(c - np.dot(c2w[:3,:4], np.array([0,0,-focal, 1.])))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
    return render_poses