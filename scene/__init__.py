#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
import numpy as np
import re
import torch
def sort_key(filename):
    # 提取文件名中的数字部分
    match = re.search(r'r_(\d+)', filename)
    if match:
        return int(match.group(1))
    else:
        return filename
def sort_key_for_camera(camera):
    # 提取文件名中的数字部分
    filename=camera.image_name
    match = re.search(r'image(\d+)', filename)
    if match:
        return int(match.group(1))
    else:
        return filename
class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, True)
        else:
            assert False, "Could not recognize scene type!"
        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

        # self.train_cameras[1.0]=self.train_cameras[1.0][:100]
        self.train_cameras[1.0].sort(key=sort_key_for_camera)
        self.next_cameras=self.train_cameras[1.0].copy()
        first=self.next_cameras.pop(0)
        self.next_cameras.append(first)
        # args.source_path="/data/byj/3dv_task/dataset/nerf_synthetic/lego"
        self.load_depth_and_normals(args.source_path)
        self.load_light_flow(args.source_path)
        # self.train_cameras[1.0]=self.train_cameras[1.0][:100]


    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    def load_depth_and_normals(self,source_path):
        source_path=os.path.join(source_path,"depth_and_normals")
        depth_path=os.path.join(source_path,"depth_npy")
        normal_path=os.path.join(source_path,"normal_npy")
        self.depths=[]
        self.normals=[]
        depth_lists=os.listdir(depth_path)
        normal_lists=os.listdir(normal_path)
        depth_lists.sort()
        normal_lists.sort()
        for i in range(len(depth_lists)):
            depth=np.load(os.path.join(depth_path,depth_lists[i]))
            normal=np.load(os.path.join(normal_path,normal_lists[i]))
            depth=torch.tensor(depth)
            # depth[depth>0.9]=0
            depth=(depth-depth.min())/(depth.max()-depth.min())
            normal=torch.tensor(normal)
            self.depths.append(depth)
            self.normals.append(normal)

    def get_depth(self):

    
        return self.depths.copy()

    def get_normal(self):
        return self.normals.copy()
    
    def get_next_camera(self):
        return self.next_cameras.copy()
    
    def load_light_flow(self,source_path):
        light_flow_path=os.path.join(source_path,"light_flow")
        self.light_flows=[]
        light_flow_lists=os.listdir(light_flow_path)
        light_flow_lists.sort()
        for i in range(len(light_flow_lists)):
            light_flow=torch.load(os.path.join(light_flow_path,light_flow_lists[i]))
            # light_flow=torch.tensor(light_flow)
            self.light_flows.append(light_flow)
    def get_light_flow(self):
        return self.light_flows.copy()
        
        