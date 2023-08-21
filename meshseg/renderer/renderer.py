# Credits to/Adopted from: https://github.com/threedle/3DHighlighter/
import torch
import numpy as np
import kaolin as kal
import matplotlib.pyplot as plt

from ..mesh.utils import get_camera_from_view2
from ..mesh.utils import device


class Renderer:
    def __init__(
        self,
        mesh="sample.obj",
        lights=torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        camera=kal.render.camera.generate_perspective_projection(np.pi / 3).to(device),
        dim=(224, 224),
    ):
        if camera is None:
            camera = kal.render.camera.generate_perspective_projection(np.pi / 3).to(
                device
            )

        self.lights = lights.unsqueeze(0).to(device)
        self.camera_projection = camera
        self.dim = dim

    def render_views(
        self,
        elev,
        azim,
        r,
        mesh,
        num_views=8,
        std=8,
        center_elev=0,
        center_azim=0,
        show=False,
        lighting=True,
        background=None,
        mask=False,
        return_views=False,
        return_mask=False,
        return_face_idx=False,
        seed=None,
    ):
        # Front view with small perturbations in viewing angle
        verts = mesh.vertices
        faces = mesh.faces
        n_faces = faces.shape[0]

        if seed is not None:
            gen = torch.Generator()
            gen = gen.manual_seed(seed)
        else:
            gen = None

        if elev is None:
            elev = torch.randn(num_views, generator=gen) * np.pi / std + center_elev
        if azim is None:
            azim = torch.randn(num_views, generator=gen) * 2 * np.pi / std + center_azim
        images = []
        masks = []
        rgb_mask = []
        faces_idx = []

        if background is not None:
            face_attributes = [
                mesh.face_attributes,  # Colors
                torch.ones((1, n_faces, 3, 1), device=device),  # hard seg. mask
            ]
        else:
            face_attributes = mesh.face_attributes

        for i in range(num_views):
            camera_transform = get_camera_from_view2(elev[i], azim[i], r=2).to(device)
            (
                face_vertices_camera,
                face_vertices_image,
                face_normals,
            ) = kal.render.mesh.prepare_vertices(
                mesh.vertices.to(device),
                mesh.faces.to(device),
                self.camera_projection,
                camera_transform=camera_transform,
            )
            image_features, soft_mask, face_idx = kal.render.mesh.dibr_rasterization(
                self.dim[1],
                self.dim[0],
                face_vertices_camera[:, :, :, -1],
                face_vertices_image,
                face_attributes,
                face_normals[:, :, -1],
            )
            masks.append(soft_mask)
            faces_idx.append(face_idx)

            if background is not None:
                image_features, mask = image_features

            image = torch.clamp(image_features, 0.0, 1.0)

            if lighting:
                image_normals = face_normals[:, face_idx].squeeze(0)
                image_lighting = kal.render.mesh.spherical_harmonic_lighting(
                    image_normals, self.lights
                ).unsqueeze(0)
                image = image * image_lighting.repeat(1, 3, 1, 1).permute(
                    0, 2, 3, 1
                ).to(device)
                image = torch.clamp(image, 0.0, 1.0)

            if background is not None:
                background_mask = torch.zeros(image.shape).to(device)
                mask = mask.squeeze(-1)
                background_idx = torch.where(mask == 0)
                assert torch.all(
                    image[background_idx] == torch.zeros(3).to(device)
                )  # Remvoe it may be taking a lot of time
                background_mask[
                    background_idx
                ] = background  # .repeat(background_idx[0].shape)
                image = torch.clamp(image + background_mask, 0.0, 1.0)
            images.append(image)

        images = torch.cat(images, dim=0).permute(0, 3, 1, 2)
        masks = torch.cat(masks, dim=0)

        if show:
            with torch.no_grad():
                fig, axs = plt.subplots(
                    1 + (num_views - 1) // 4, min(4, num_views), figsize=(89.6, 22.4)
                )
                for i in range(num_views):
                    if num_views == 1:
                        ax = axs
                    elif num_views <= 4:
                        ax = axs[i]
                    else:
                        ax = axs[i // 4, i % 4]
                    ax.imshow(images[i].permute(1, 2, 0).cpu().numpy())
                plt.show()

        if return_views == True:
            if return_mask == True:
                assert return_face_idx == True
                return images, elev, azim, masks, faces_idx
            else:
                return images, elev, azim
        else:
            return images
