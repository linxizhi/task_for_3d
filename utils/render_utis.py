import numpy as np
import dptr.gs as gs
import torch
from scene import Scene, GaussianModel
from PIL import Image
import sys
sys.path.insert(0,'/data/byj/3dv_task/gaussian-splatting/RAFT')
from RAFT.core.utils import flow_viz
def my_render(xyz,features, scaling,rotation,opacity,normals,viewpoint_cam,next_viewpoint_cam):        
        direction = (xyz.cuda() -
                     viewpoint_cam.camera_center.repeat(xyz.shape[0], 1).cuda())
        direction = direction / direction.norm(dim=1, keepdim=True)
        rgb = gs.compute_sh(features, 3, direction)

        height,width=viewpoint_cam.image_height,viewpoint_cam.image_width
        world_view_transform = viewpoint_cam.world_view_transform.cuda()
        full_proj_transform = viewpoint_cam.full_proj_transform.cuda()
        
        intr = torch.Tensor([
            width / (2 * np.tan(viewpoint_cam.FoVx * 0.5)),
            height / (2 * np.tan(viewpoint_cam.FoVy * 0.5)),
            float(width) / 2,
            float(height) / 2]).cuda().float()

        (uv, depth) = gs.project_point(
            xyz,
            world_view_transform,
            full_proj_transform,
            intr, width, height)
        (uv_next,depeth_next)=gs.project_point(
            xyz,
            next_viewpoint_cam.world_view_transform.cuda(),
            next_viewpoint_cam.full_proj_transform.cuda(),
            intr, width, height
        )
        
        delta_uv=uv_next-uv
        
        
        visible = depth != 0
        cov3d = gs.compute_cov3d(scaling, rotation, visible)
        (conic, radius, tiles_touched) = gs.ewa_project(
            xyz,
            cov3d,
            world_view_transform,
            intr,
            uv,
            width, 
            height,
            visible
        )
        (gaussian_ids_sorted, tile_range) = gs.sort_gaussian(
            uv, depth, width, height, radius, tiles_touched
        )
        features=[rgb,depth,normals,delta_uv]
        features=torch.cat(features,dim=-1)
        ndc = torch.zeros_like(uv, requires_grad=True)
        try:
            ndc.retain_grad()
        except:
            raise ValueError("ndc does not have grad")
        rendered_features = gs.alpha_blending(
            uv, conic, opacity, features,
            gaussian_ids_sorted, tile_range,0.0, width, height  , ndc
        )
        image=rendered_features[0:3,...]
        depth=rendered_features[3:4,...]
        normals=rendered_features[4:7,...]
        light_flow=rendered_features[7:9,...]
        eps=1e-8
        normals[-1,...]+=eps
        xx=torch.norm(normals,dim=0,keepdim=True)
        normals=normals/xx
        visibility_filter=radius>0
        radii=radius
        viewspace_point_tensor=ndc
        return visibility_filter,radii,viewspace_point_tensor,image,depth,normals,light_flow
def render_for_predict(xyz,features, scaling,rotation,opacity,normals,viewpoint_cam,next_viewpoint_cam):        
        direction = (xyz.cuda() -
                     viewpoint_cam.camera_center.repeat(xyz.shape[0], 1).cuda())
        direction = direction / direction.norm(dim=1, keepdim=True)
        rgb = gs.compute_sh(features, 3, direction)

        height,width=viewpoint_cam.image_height,viewpoint_cam.image_width
        world_view_transform = viewpoint_cam.world_view_transform.cuda()
        full_proj_transform = viewpoint_cam.full_proj_transform.cuda()
        
        intr = torch.Tensor([
            width / (2 * np.tan(viewpoint_cam.FoVx * 0.5)),
            height / (2 * np.tan(viewpoint_cam.FoVy * 0.5)),
            float(width) / 2,
            float(height) / 2]).cuda().float()

        (uv, depth) = gs.project_point(
            xyz,
            world_view_transform,
            full_proj_transform,
            intr, width, height)

        
        (uv_next,depeth_next)=gs.project_point(
            xyz,
            next_viewpoint_cam.world_view_transform.cuda(),
            next_viewpoint_cam.full_proj_transform.cuda(),
            intr, width, height
        )
        
        delta_uv=uv_next-uv
        visible = depth != 0
        cov3d = gs.compute_cov3d(scaling, rotation, visible)
        (conic, radius, tiles_touched) = gs.ewa_project(
            xyz,
            cov3d,
            world_view_transform,
            intr,
            uv,
            width, 
            height,
            visible
        )
        (gaussian_ids_sorted, tile_range) = gs.sort_gaussian(
            uv, depth, width, height, radius, tiles_touched
        )
        features=[rgb,depth,normals,delta_uv]
        features=torch.cat(features,dim=-1)
        
        ndc = torch.zeros_like(uv, requires_grad=True)
        try:
            ndc.retain_grad()
        except:
            raise ValueError("ndc does not have grad")
        rendered_features = gs.alpha_blending(
            uv, conic, opacity, features,
            gaussian_ids_sorted, tile_range,0.0, width, height  , ndc
        )

        image=rendered_features[0:3,...]
        depth=rendered_features[3:4,...]
        normals=rendered_features[4:7,...]
        light_flow=rendered_features[7:9,...]
        eps=1e-8
        normals[-1,...]+=eps
        xx=torch.norm(normals,dim=0,keepdim=True)
        normals=normals/xx
        visibility_filter=radius>0
        radii=radius
        viewspace_point_tensor=ndc
        out={
            'visibility_filter':visibility_filter,
            'radii':radii,
            'viewspace_point_tensor':viewspace_point_tensor,
            'image':image,
            'depth':depth,
            'normals':normals,
            "light_flow":light_flow
        }
        return out



def return_all_need(gaussians:GaussianModel):
    return gaussians.get_xyz,gaussians.get_scaling,gaussians.get_rotation,gaussians.get_opacity,gaussians.get_features,gaussians.get_normals()



def depth2normal(depth):
    depth=depth.squeeze()
    w,h=depth.shape
    dx=-(depth[2:h,1:h-1]-depth[0:h-2,1:h-1])*0.5
    dy=-(depth[1:h-1,2:h]-depth[1:h-1,0:h-2])*0.5
    dz=torch.ones((w-2,h-2),device=depth.device)
    dl = torch.sqrt(dx * dx + dy * dy + dz * dz)
    dx = dx / dl * 0.5 + 0.5
    dy = dy / dl * 0.5 + 0.5
    dz = dz / dl * 0.5 + 0.5
    return torch.stack([dy,dx,dz],dim=0)

def process_normals(normals:torch.Tensor,viewpoint_cam:Scene,xyz:torch.Tensor):
    direction = (xyz.cuda() -
                     viewpoint_cam.camera_center.repeat(xyz.shape[0], 1).cuda())
    direction = direction / direction.norm(dim=1, keepdim=True)
    dot_for_judge=torch.sum(direction*normals,dim=-1)
    normals[dot_for_judge<0]=-normals[dot_for_judge<0]
    w2c=torch.tensor(viewpoint_cam.R).cuda().float()
    normals_image=normals@w2c.T

    return normals_image

def viz(flo,idx):
    flo=flo.detach()
    flo=flo.squeeze()
    flo = flo.permute(1,2,0).cpu().numpy()
    flo = flow_viz.flow_to_image(flo)
    image=Image.fromarray(flo)
    image.save(f'png_out/{idx}.jpg')