import torch
import trimesh
import numpy as np
import scipy.optimize
from tqdm.auto import tqdm
import potpourri3d as pp3d

from copy import deepcopy
from collections import Counter
from collections import defaultdict
from scipy.spatial.distance import cdist

from ..models.GLIP.glip import GLIPModel


class BaseMeshSegmentor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.rendered_images = None
        self.rendered_images_face_ids = None
        self.prompts = None
        self.mesh = None

    def color_mesh_and_save(self, face_cls, colors, output_filename="test.obj"):
        # Now color the facees according to the predicted class
        face_colors = torch.zeros((len(self.mesh.faces), 4), dtype=torch.uint8)

        for i in range(len(self.mesh.faces)):
            face_colors[i][0:3] = torch.tensor(colors[face_cls[i]]) * 255
            face_colors[i][-1] = 255

        # Create trimesh
        scene = trimesh.Scene()
        output_mesh = trimesh.Trimesh(
            vertices=self.mesh.vertices.cpu().numpy(),
            faces=self.mesh.faces.cpu().numpy(),
        )
        output_mesh.visual.face_colors = face_colors.cpu().numpy()
        scene.add_geometry(output_mesh, node_name="output")
        _ = scene.export(output_filename)


class BaseDetMeshSegmentor(BaseMeshSegmentor):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.bbox_predictions = None

    def set_mesh(self, mesh):
        self.mesh = mesh

    def set_prompts(self, prompts):
        self.prompts = prompts

    def set_rendered_views(
        self, rendered_images: torch.Tensor, images_face_ids: torch.Tensor
    ):
        self.rendered_images = rendered_images
        self.rendered_images_face_ids = images_face_ids

    def get_included_face_ids(self, pixel_face_ids, bbox_cor, face_counter):
        relevant_face_ids = defaultdict(int)
        non_relevant_face_ids = defaultdict(int)

        bbox_cor = bbox_cor.to(torch.int64)
        (col_min, row_min), (col_max, row_max) = (
            bbox_cor[:2].tolist(),
            bbox_cor[2:].tolist(),
        )

        for col in range(col_min, col_max):
            for row in range(row_min, row_max):
                face_id = pixel_face_ids[row][col].item()
                if face_id == -1:
                    continue
                relevant_face_ids[face_id] += 1

        for k, _ in face_counter.items():
            if k == -1:
                continue
            if k not in relevant_face_ids:
                non_relevant_face_ids[k] += 1

        return relevant_face_ids, non_relevant_face_ids

    def process_box_predictions(self, prompt, preds, rendering_face_ids):
        # Initialize a score vector for each face in the mesh for the given region prompt (e.g. "the leg of a person").
        face_view_prompt_score = np.zeros((len(self.mesh.faces)))
        face_view_freq = np.zeros((len(self.mesh.faces)))

        pred_bboxes = preds[0][1]
        pred_bboxes_cor = preds[0][1].bbox
        n_boxes = len(
            pred_bboxes
        )  # The number of predcited bounding boxes for the given prompt (e.g., the leg of a person).

        face_counter = rendering_face_ids.flatten().cpu().int().numpy()
        face_counter = face_counter[face_counter != -1]
        face_counter = Counter(list(face_counter))

        for i in range(n_boxes):
            if pred_bboxes.get_field("labels")[i].item() != 1 and "of" in prompt:
                continue

            confidence_score = pred_bboxes.get_field("scores")[i].item()

            # Get the included face ids inside this bounding box
            included_face_ids, not_included_face_ids = self.get_included_face_ids(
                rendering_face_ids, pred_bboxes_cor[i], face_counter
            )

            for k, v in included_face_ids.items():
                face_view_prompt_score[k] += v * confidence_score

        return face_view_prompt_score, face_view_freq

    def __call__(self):
        pass


class GLIPMeshSegmenter(BaseDetMeshSegmentor):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.glip_model = GLIPModel()

    def predict_face_cls(self):
        # Get the bounding boxes predictions for the given prompts
        self.predict_bboxes()
        assert self.bbox_predictions is not None

        face_cls = np.zeros((len(self.mesh.faces), len(self.prompts)))
        face_freq = np.zeros((len(self.mesh.faces), len(self.prompts)))

        # Looping over the views
        for i, view in tqdm(enumerate(self.rendered_images)):
            for j, prompt in enumerate(self.prompts):
                (
                    face_view_prompt_score,
                    face_view_prompt_freq,
                ) = self.process_box_predictions(
                    prompt,
                    self.bbox_predictions[i][prompt],
                    self.rendered_images_face_ids[i].squeeze(0),
                )
                face_cls[:, j] += face_view_prompt_score
                face_freq[:, j] += face_view_prompt_freq

        return face_cls, face_freq

    def predict_bboxes(self):
        # Generate GLIP predictions for every rendered image
        print("Feeding the views to GLIP...")
        self.bbox_predictions = []
        for i in range(len(self.rendered_images)):
            self.bbox_predictions.append({})

            img = self.rendered_images[i].permute([1, 2, 0]) * 256.0
            img = img.to(torch.uint8)

            for p in self.prompts:
                res = self.glip_model.predict(img.cpu().numpy(), p)
                self.bbox_predictions[i][p] = (res, self.glip_model.model.entities)

    def __call__(self):
        assert self.glip_model is not None
        assert self.rendered_images is not None

        return self.predict_face_cls()


class SATR(GLIPMeshSegmenter):
    def __init__(self, cfg):
        super().__init__(cfg)

        (
            self.face_adjacency,
            self.faces_distance_factors,
            self.center_vert_id,
            self.face_visibilty_ratios,
        ) = (None, None, None, None)

    def set_mesh(self, mesh):
        super().set_mesh(mesh)

        self.get_faces_neighborhood()
        self.compute_vertices_pairwise_dist()

    def get_faces_neighborhood(self):
        n = self.cfg.satr.face_smoothing_n_ring

        m = trimesh.Trimesh(
            vertices=self.mesh.vertices.cpu().numpy(),
            faces=self.mesh.faces.cpu().numpy(),
        )

        faces_adjacency = defaultdict(set)

        for el in m.face_adjacency:
            u, v = el
            faces_adjacency[u].add(v)
            faces_adjacency[v].add(u)

        c = []
        faces_adjacency_res = deepcopy(faces_adjacency)

        for k in faces_adjacency:
            for i in range(n - 1):
                start = deepcopy(faces_adjacency_res[k])
                end = set(deepcopy(faces_adjacency_res[k]))
                for f in start:
                    end.update(faces_adjacency[f])
                faces_adjacency_res[k] = end
            c.append(len(faces_adjacency_res[k]))

        self.face_adjacency = faces_adjacency_res

    def compute_faces_to_capital_face_distances(self, faces_dict):
        visible_faces_ids = torch.tensor(sorted(list(faces_dict.keys()))).int().cuda()
        visible_faces = torch.index_select(self.mesh.faces, 0, visible_faces_ids)
        visible_vertices_ids = visible_faces.flatten().unique()

        # Get the mean vertex position of all the faces visible
        v1 = torch.index_select(self.mesh.vertices, 0, visible_faces[:, 0])
        v2 = torch.index_select(self.mesh.vertices, 0, visible_faces[:, 1])
        v3 = torch.index_select(self.mesh.vertices, 0, visible_faces[:, 2])

        visible_face_areas = torch.index_select(
            self.mesh.face_areas, 0, visible_faces_ids
        ).view(len(visible_faces_ids), 1)

        faces_center = ((v1 + v2 + v3) / 3.0) * visible_face_areas

        capital_face_center = faces_center.sum(dim=0) / visible_face_areas.sum(dim=0)

        # Find the nearest vertex to the capital_face_center
        distances = cdist(
            self.mesh.vertices[visible_vertices_ids].cpu().numpy(),
            capital_face_center.cpu().view(1, 3).numpy(),
        )

        nearest_vert_ind = np.argmin(distances[:, 0])
        nearest_vert_id = visible_vertices_ids[nearest_vert_ind]

        # Now compute the distance from this vertex (representing the capital face on the mesh) 
        # to each face found in faces_dict
        faces_distance = {}
        distances = []

        for k, _ in faces_dict.items():
            face_vert = self.mesh.faces[k][0].cpu().numpy()
            distances.append(self.vertices_distances[face_vert, nearest_vert_id])
            faces_distance[k] = self.vertices_distances[face_vert, nearest_vert_id]

        distances = np.array(distances)
        mean = np.mean(distances)
        std = np.std(distances)
        g = scipy.stats.norm(mean, std)

        faces_distance_factor = {}
        for k, v in faces_dict.items():
            faces_distance_factor[k] = g.pdf(faces_distance[k])

        return faces_distance_factor, nearest_vert_id

    def compute_vertices_pairwise_dist(self):
        n_vertices = len(self.mesh.vertices)

        solver = pp3d.MeshHeatMethodDistanceSolver(
            self.mesh.vertices.cpu().numpy(), self.mesh.faces.cpu().numpy()
        )

        self.vertices_distances = np.zeros((n_vertices, n_vertices))
        for i in range(n_vertices):
            x = solver.compute_distance(i)
            self.vertices_distances[:, i] = x

    def preprocessing_step_reweighting_factors(self, included_face_ids):
        if self.cfg.satr.gaussian_reweighting:
            (
                self.faces_distance_factors,
                self.center_vert_id,
            ) = self.compute_faces_to_capital_face_distances(included_face_ids)

        if self.cfg.satr.face_smoothing:
            self.face_visibilty_ratios = self.compute_face_visibilty_ratio(
                included_face_ids
            )

    def compute_reweighting_factors(self, included_face_ids):
        ret = {"faces_distance_factor": {}, "face_visibilty_ratio": {}}

        for k, _ in included_face_ids.items():
            ret["faces_distance_factor"][k] = (
                self.faces_distance_factors[k]
                if self.cfg.satr.gaussian_reweighting
                else 1.0
            )
            ret["face_visibilty_ratio"][k] = (
                self.face_visibilty_ratios[k] if self.cfg.satr.face_smoothing else 1.0
            )

        return ret

    def compute_face_visibilty_ratio(self, relevant_face_ids):
        face_neighborhood = self.face_adjacency

        relevant_face_scores = {}

        for k, v in relevant_face_ids.items():
            score = 0
            for f in face_neighborhood[k]:
                score += f in relevant_face_ids
            score /= len(face_neighborhood[k]) + 0.00001
            relevant_face_scores[k] = score

        return relevant_face_scores

    def process_box_predictions(self, prompt, preds, rendering_face_ids):
        # Initialize a score vector for each face in the mesh for the given region prompt (e.g. "the leg of a person").
        face_view_prompt_score = np.zeros((len(self.mesh.faces)))
        face_view_freq = np.zeros((len(self.mesh.faces)))

        pred_bboxes = preds[0][1]
        pred_bboxes_cor = preds[0][1].bbox
        n_boxes = len(
            pred_bboxes
        )  # The number of predcited bounding boxes for the given prompt (e.g., the leg of a person).

        face_counter = rendering_face_ids.flatten().cpu().int().numpy()
        face_counter = face_counter[face_counter != -1]
        face_counter = Counter(list(face_counter))

        for i in range(n_boxes):
            if pred_bboxes.get_field("labels")[i].item() != 1 and "of" in prompt:
                continue

            confidence_score = pred_bboxes.get_field("scores")[i].item()

            # Get the included face ids inside this bounding box
            included_face_ids, not_included_face_ids = self.get_included_face_ids(
                rendering_face_ids, pred_bboxes_cor[i], face_counter
            )

            # Compute the reweighting factors
            self.preprocessing_step_reweighting_factors(
                included_face_ids,
            )
            reweighting_factors = self.compute_reweighting_factors(included_face_ids)

            for k, v in included_face_ids.items():
                final_factor = 1.0

                for f in list(reweighting_factors.values()):
                    final_factor *= f[k]

                face_view_prompt_score[k] += v * confidence_score * final_factor

        return face_view_prompt_score, face_view_freq
