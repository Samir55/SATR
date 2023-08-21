# Credits to https://github.com/threedle/3DHighlighter/blob/main/utils.py
import torch
import kaolin.ops.mesh
import kaolin as kal
from . import MeshNormalizer

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

def get_camera_from_view2(elev, azim, r=3.0):
    x = r * torch.cos(elev) * torch.cos(azim)
    y = r * torch.sin(elev)
    z = r * torch.cos(elev) * torch.sin(azim)
    # print(elev,azim,x,y,z)

    pos = torch.tensor([x, y, z]).unsqueeze(0)
    look_at = -pos
    direction = torch.tensor([0.0, 1.0, 0.0]).unsqueeze(0)

    camera_proj = kal.render.camera.generate_transformation_matrix(pos, look_at, direction)
    return camera_proj

def get_texture_map_from_color(mesh, color, H=224, W=224):
    texture_map = torch.zeros(1, H, W, 3).to(device)
    texture_map[:, :, :] = color
    return texture_map.permute(0, 3, 1, 2)


def get_face_attributes_from_color(mesh, color):
    num_faces = mesh.faces.shape[0]
    face_attributes = torch.zeros(1, num_faces, 3, 3).to(device)
    face_attributes[:, :, :] = color
    return face_attributes

# mesh coloring helpers
def color_mesh(pred_class, sampled_mesh, colors):
    pred_rgb = segment2rgb(pred_class, colors)
    sampled_mesh.face_attributes = kaolin.ops.mesh.index_vertices_by_faces(
        pred_rgb.unsqueeze(0), sampled_mesh.faces
    )
    MeshNormalizer(
        sampled_mesh
    )()  # This is weird why it is called, it should be removed I think FIXME @ahmed


def segment2rgb(pred_class, colors):
    pred_rgb = torch.zeros(pred_class.shape[0], 3).to(device)
    for class_idx, color in enumerate(colors):
        pred_rgb += torch.matmul(
            pred_class[:, class_idx].unsqueeze(1), color.unsqueeze(0)
        )

    return pred_rgb
