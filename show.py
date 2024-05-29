import os
import numpy as np
import torch
from PIL import Image
import matplotlib
def load_depth_and_normals(source_path):
    source_path=os.path.join(source_path,"depth_and_normals")
    depth_path=os.path.join(source_path,"depth_npy")
    normal_path=os.path.join(source_path,"normal_npy")
    depths=[]
    normals=[]
    depth_lists=os.listdir(depth_path)
    normal_lists=os.listdir(normal_path)
    depth_lists.sort()
    normal_lists.sort()
    for i in range(len(depth_lists)):
        depth=np.load(os.path.join(depth_path,depth_lists[i]))
        normal=np.load(os.path.join(normal_path,normal_lists[i]))
        depth=torch.tensor(depth)

        depth=(depth-depth.min())/(depth.max()-depth.min())
        depth[depth>0.9]=0
        normal=torch.tensor(normal)
        depths.append(depth)
        normals.append(normal)
    return depths,normals
def save_image(data,index=0):
    image = torch.clamp(data, 0.0, 1.0)
    image=image.squeeze()
    np_image=(image.cpu().detach().numpy()*255).astype(np.uint8)
    filename=f"output{index}.png"
    if image.dim()==3 and image.shape[0]==3:
        np_image=np.transpose(np_image,(1,2,0))
    pil_image = Image.fromarray(np_image)
    
    # 保存图像到文件
    pil_image.save(filename)
    return image

def colorize_depth_maps(
    depth_map, min_depth, max_depth, cmap="Spectral", valid_mask=None
):
    """
    Colorize depth maps.
    """
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
if __name__=="__main__":
    source_path='/data/byj/3dv_task/dataset/nerf_synthetic/lego'
    depths,normals= load_depth_and_normals(source_path)
    depth_pred=depths[0]
    color_map="Spectral"
    depth_colored = colorize_depth_maps(
            depth_pred, 0, 1, cmap=color_map
        ).squeeze()  # [3, H, W], value in (0, 1)
    save_image(depth_colored,10)
    save_image(depth_pred,11)
    a=0