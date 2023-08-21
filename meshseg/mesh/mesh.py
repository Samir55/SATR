# Credits to https://github.com/threedle/3DHighlighter/blob/main/mesh.py
import kaolin as kal
import torch
from .utils import device
from .utils import *
import copy
import numpy as np
import PIL


class Mesh:
    def __init__(self, obj_path, color=torch.tensor([0.0, 0.0, 1.0])):
        if ".obj" in obj_path:
            mesh = kal.io.obj.import_mesh(
                obj_path,
                with_normals=True,
                with_materials=False,
            )
        elif ".off" in obj_path:
            mesh = kal.io.off.import_mesh(obj_path)
        else:
            raise ValueError(f"{obj_path} extension not implemented in mesh reader.")
        self.vertices = mesh.vertices.to(device)
        self.faces = mesh.faces.to(device)
        self.face_areas = kal.ops.mesh.face_areas(
            self.vertices.unsqueeze(0), self.faces
        ).view(len(mesh.faces))
        self.vertex_normals = None
        self.face_normals = None
        self.texture_map = None
        self.face_uvs = None
        if ".obj" in obj_path:
            if mesh.vertex_normals is not None:
                self.vertex_normals = mesh.vertex_normals.to(device).float()

                # Normalize
                self.vertex_normals = torch.nn.functional.normalize(self.vertex_normals)

            if mesh.face_normals is not None:
                self.face_normals = mesh.face_normals.to(device).float()

                # Normalize
                self.face_normals = torch.nn.functional.normalize(self.face_normals)

            # if/

        self.set_mesh_color(color)
        # self.face_attributes =

    def standardize_mesh(self, inplace=False):
        mesh = self if inplace else copy.deepcopy(self)
        return standardize_mesh(mesh)

    def normalize_mesh(self, inplace=False):
        mesh = self if inplace else copy.deepcopy(self)
        return normalize_mesh(mesh)

    def update_vertex(self, verts, inplace=False):
        mesh = self if inplace else copy.deepcopy(self)
        mesh.vertices = verts
        return mesh

    def set_mesh_color(self, color):
        self.texture_map = get_texture_map_from_color(self, color)
        self.face_attributes = get_face_attributes_from_color(self, color)

    def set_image_texture(self, texture_map, inplace=True):
        mesh = self if inplace else copy.deepcopy(self)

        if isinstance(texture_map, str):
            texture_map = PIL.Image.open(texture_map)
            texture_map = np.array(texture_map, dtype=np.float) / 255.0
            texture_map = (
                torch.tensor(texture_map, dtype=torch.float)
                .to(device)
                .permute(2, 0, 1)
                .unsqueeze(0)
            )

        mesh.texture_map = texture_map
        return mesh

    def divide(self, inplace=True):
        mesh = self if inplace else copy.deepcopy(self)
        (
            new_vertices,
            new_faces,
            new_vertex_normals,
            new_face_uvs,
            new_face_normals,
        ) = add_vertices(mesh)
        mesh.vertices = new_vertices
        mesh.faces = new_faces
        mesh.vertex_normals = new_vertex_normals
        mesh.face_uvs = new_face_uvs
        mesh.face_normals = new_face_normals
        return mesh

    def divide_with_labels(self, face_label_map, inplace=True):
        mesh = self if inplace else copy.deepcopy(self)
        (
            new_vertices,
            new_faces,
            new_vertex_normals,
            new_face_label_map,
        ) = add_vertices_with_labels(mesh, face_label_map)
        mesh.vertices = new_vertices
        mesh.faces = new_faces
        mesh.vertex_normals = new_vertex_normals
        return mesh, new_face_label_map

    def export(self, file, extension="obj", color=None):
        with open(file, "w+") as f:
            if extension == "obj":
                for vi, v in enumerate(self.vertices):
                    if color is None:
                        f.write("v %f %f %f\n" % (v[0], v[1], v[2]))
                    else:
                        f.write(
                            "v %f %f %f %f %f %f\n"
                            % (
                                v[0],
                                v[1],
                                v[2],
                                color[vi][0],
                                color[vi][1],
                                color[vi][2],
                            )
                        )
                    if self.vertex_normals is not None:
                        f.write(
                            "vn %f %f %f\n"
                            % (
                                self.vertex_normals[vi, 0],
                                self.vertex_normals[vi, 1],
                                self.vertex_normals[vi, 2],
                            )
                        )
                for face in self.faces:
                    f.write("f %d %d %d\n" % (face[0] + 1, face[1] + 1, face[2] + 1))
            elif extension == "ply":
                f.write("ply\nformat ascii 1.0\n")
                f.write("element vertex {}\n".format(self.vertices.shape[0]))
                f.write("property float32 x\nproperty float32 y\nproperty float32 z\n")
                f.write(
                    "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                )
                f.write("element face {}\n".format(self.faces.shape[0]))
                f.write("property list uint8 int32 vertex_index\n")
                f.write("end_header\n")
                for vi, v in enumerate(self.vertices):
                    f.write(
                        "%f %f %f %d %d %d\n"
                        % (v[0], v[1], v[2], color[vi][0], color[vi][1], color[vi][2])
                    )
                for face in self.faces:
                    f.write("3 %d %d %d\n" % (face[0], face[1], face[2]))
            else:
                print("error -- mesh.py: incorrect extension provdied in mesh.export()")
