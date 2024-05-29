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

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.render_utis import render_for_predict,return_all_need,process_normals
import sys
sys.path.insert(0,'/data/byj/3dv_task/gaussian-splatting/RAFT')
from PIL import Image
from RAFT.core.utils import flow_viz
import matplotlib
import numpy as np
def viz(flo,path):
    flo=flo.detach()
    flo=flo.squeeze()
    flo = flo.permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    image=Image.fromarray(flo)
    # filename=idx
    image.save(path)
def save_image(data,path):
    image = torch.clamp(data, 0.0, 1.0)
    image=image.squeeze()
    np_image=(image.cpu().detach().numpy()*255).astype(np.uint8)

    if image.dim()==3 and image.shape[0]==3:
        np_image=np.transpose(np_image,(1,2,0))
    pil_image = Image.fromarray(np_image)
    
    # 保存图像到文件
    pil_image.save(path)
    return image

def colorize_depth_maps(
    depth_map, min_depth, max_depth, cmap="Spectral", valid_mask=None
):
    """
    Colorize depth maps.
    """
    depth_map=depth_map.cpu()
    min_depth=min_depth.cpu().detach().numpy()
    max_depth=max_depth.cpu().detach().numpy()
    assert len(depth_map.shape) >= 2, "Invalid dimension"

    if isinstance(depth_map, torch.Tensor):
        depth = depth_map.detach().clone().squeeze().numpy()
    elif isinstance(depth_map, np.ndarray):
        depth = depth_map.copy().squeeze()
    # reshape to [ (B,) H, W ]
    if depth.ndim < 3:
        depth = depth[np.newaxis, :, :]

    # colorize
    cm = matplotlib.colormaps[cmap]
    depth = ((depth - min_depth) / (max_depth - min_depth)).clip(0, 1)
    img_colored_np = cm(depth, bytes=False)[:, :, :, 0:3]  # value from 0 to 1
    img_colored_np = np.rollaxis(img_colored_np, 3, 1)

    if valid_mask is not None:
        if isinstance(depth_map, torch.Tensor):
            valid_mask = valid_mask.detach().numpy()
        valid_mask = valid_mask.squeeze()  # [H, W] or [B, H, W]
        if valid_mask.ndim < 3:
            valid_mask = valid_mask[np.newaxis, np.newaxis, :, :]
        else:
            valid_mask = valid_mask[:, np.newaxis, :, :]
        valid_mask = np.repeat(valid_mask, 3, axis=1)
        img_colored_np[~valid_mask] = 0

    if isinstance(depth_map, torch.Tensor):
        img_colored = torch.from_numpy(img_colored_np).float()
    elif isinstance(depth_map, np.ndarray):
        img_colored = img_colored_np

    return img_colored

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    depths_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depths")
    depths_path_np = os.path.join(model_path, name, "ours_{}".format(iteration), "depths_np")
    
    normals_path = os.path.join(model_path, name, "ours_{}".format(iteration), "normals")
    light_flow_path=os.path.join(model_path, name, "ours_{}".format(iteration), "light_flow")
    depths_rgb_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depths_rgb")
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(normals_path, exist_ok=True)
    makedirs(depths_path, exist_ok=True)
    makedirs(depths_path_np, exist_ok=True)
    makedirs(light_flow_path, exist_ok=True)
    makedirs(depths_rgb_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        next_view=views[(idx+1)%len(views)]
        # rendering = render(view, gaussians, pipeline, background)["render"]
        xyz,scaling,rotation,opacity,features,normals=return_all_need(gaussians)
        normals_of_camera=process_normals(normals,view,xyz)
        out=render_for_predict(xyz,features, scaling,rotation,opacity,normals_of_camera,view,next_view)
        image=out['image']
        depth=(out['depth']-out["depth"].min()) /(out['depth'].max()-out['depth'].min())
        depth_rgb=colorize_depth_maps(out['depth'],out['depth'].min(),out['depth'].max())
        normals=(out['normals']+1)/2
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(image, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(normals, os.path.join(normals_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(depth, os.path.join(depths_path, '{0:05d}'.format(idx) + ".png"))
        viz(out['light_flow'],os.path.join(light_flow_path, '{0:05d}'.format(idx) + ".png"))
        save_image(depth_rgb,path=os.path.join(depths_rgb_path, '{0:05d}'.format(idx) + ".png"))
        
def myrender_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))




def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)