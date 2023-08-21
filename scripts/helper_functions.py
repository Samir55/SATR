import os
import time
import json
import torch
import numpy as np
import kaolin.ops.mesh
import os.path as osp
from ast import literal_eval
from omegaconf import OmegaConf

from meshseg.mesh import MeshNormalizer
from meshseg.mesh import Mesh
from meshseg.renderer.renderer import Renderer
from meshseg.methods.segmentors import *

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")


# Now we can evaluate now the baseline vs SATR our proposed method
def prepare_seg_classes(input_prompt, input_mesh_class):
    prompt = str(input_prompt)
    ps = literal_eval(prompt)
    part_names = sorted(ps)
    prompts = sorted(ps)
    for i, p in enumerate(prompts):
        prompts[i] = f"the {p} of a {input_mesh_class}."
    print(prompts)

    ind = 0
    cls_name_to_id = {}
    cls_id_to_name = {}

    for i, p in enumerate(part_names):
        cls_name_to_id[p] = ind
        cls_id_to_name[ind] = p

        ind += 1

    cls_name_to_id["unknown"] = ind
    cls_id_to_name[ind] = "unknown"

    print(cls_name_to_id)

    return prompts, part_names, cls_name_to_id, cls_id_to_name


def segment(
    config_path,
    mesh_name,
    output_dir,
    input_prompt=None,
    mesh_class=None,
):
    ss = time.time()

    config = OmegaConf.load(config_path)

    mesh_path = os.path.join(config.dataset_dir, mesh_name)
    print(mesh_path)
    assert os.path.isfile(mesh_path)

    os.makedirs(output_dir, exist_ok=True)

    # Save the config in the output folder
    with open(os.path.join(output_dir, "config.yaml"), "w") as fout:
        OmegaConf.save(config, fout)

    # Get the prompts
    if input_prompt is None:
        assert config.region_names is not None and type(config.region_names) == str
        assert config.object_class is not None
        input_prompt = config.region_names
        mesh_class = config.object_class

    prompts, part_names, cls_name_to_id, cls_id_to_name = prepare_seg_classes(
        input_prompt, mesh_class
    )

    # Read the mesh
    print(f"Reading the mesh...")
    mesh = Mesh(mesh_path)
    print(
        f"Reading the mesh with path: {mesh_path}\n\thaving {mesh.faces.shape[0]} faces and {mesh.vertices.shape[0]} vertices"
    )

    # normalize mesh in unit sphere
    _ = MeshNormalizer(mesh)()

    # Put default grey color to the mesh
    if "color" not in config:
        mesh.face_attributes = kaolin.ops.mesh.index_vertices_by_faces(
            torch.ones(1, len(mesh.vertices), 3).to(device)
            * torch.tensor([0.5, 0.5, 0.5]).unsqueeze(0).unsqueeze(0).to(device),
            mesh.faces,
        )
    else:
        mesh.face_attributes = kaolin.ops.mesh.index_vertices_by_faces(
            torch.ones(1, len(mesh.vertices), 3).cuda()
            * torch.tensor(config.color).unsqueeze(0).unsqueeze(0).cuda(),
            mesh.faces,
        )

    # Create the renderer
    print(f"Creating the renderer...")
    render = Renderer(dim=(config.camera.render_res, config.camera.render_res))

    # Initialize Background
    background = torch.tensor([0.0, 0.0, 0.0]).to(device)

    with torch.no_grad():
        print(f"Rendering the views...")
        elev = None
        azim = None
        rendered_images, elev, azim, _, faces_idx = render.render_views(
            elev,
            azim,
            2,
            mesh,
            num_views=config.camera.n_views,
            show=False,
            center_azim=config.camera.frontview_center[0],
            center_elev=config.camera.frontview_center[1],
            std=config.camera.frontview_std,
            return_views=True,
            return_mask=True,
            return_face_idx=True,
            lighting=True,
            background=background,
            seed=2023,
        )
        print(f"Rendering the views...done")

    #
    # Segmentation
    #
    # Create the segmenter
    segmenter = SATR(config)
    segmenter.set_mesh(mesh)
    segmenter.set_prompts(prompts)
    segmenter.set_rendered_views(rendered_images, faces_idx)

    # Segment
    predictions, _ = segmenter()
    predictions = torch.tensor(predictions)

    # Save the predictions
    np.save(os.path.join(output_dir, "raw_face_preds.npy"), predictions.cpu().numpy())

    faces_not_assigned = torch.where(torch.sum(predictions, axis=-1) < 0.0001)[0]
    predictions_cls = predictions.argmax(axis=-1)
    predictions_cls[faces_not_assigned] = len(prompts)

    cols = [
        [247 / 255, 165 / 255, 94 / 255.0],
        [247 / 255.0, 94 / 255.0, 165 / 255.0],
        [94 / 255.0, 165 / 255.0, 247 / 255.0],
        [100 / 255.0, 0, 100 / 255.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [0.0, 0.5, 1.0],
        [0.2, 0.2, 0.9],
        [0.1, 0.5, 0.9],
        [0.9, 0.2, 0.9],
        [0.9, 0.4, 0.1],
        [0.5, 0.0, 0.9],
        [0.1, 0.9, 0.5],
    ]
    cols = cols[: len(part_names)]
    cols += [[0.5, 0.5, 0.5]]  # for the unknown class

    segmenter.color_mesh_and_save(
        predictions_cls, cols, os.path.join(output_dir, "out_mesh_seg.obj")
    )

    # The predictions now are saved as strings
    with open(os.path.join(output_dir, "face_preds.json"), "w") as fout:
        faces_cls = []
        for el in list(predictions_cls.cpu().numpy().astype(int)):
            faces_cls.append(cls_id_to_name[int(el)])
        json.dump(faces_cls, fout)

    with open(osp.join(output_dir, "vert_preds.json"), "w") as fout:
        vertices_cls = [""] * len(mesh.vertices)
        for i_f, f_vs in enumerate(mesh.faces):
            for v in f_vs:
                vertices_cls[v] = cls_id_to_name[int(predictions_cls[i_f])]

        json.dump(vertices_cls, fout)

    with open(os.path.join(output_dir, "elapsed_time.txt"), "w") as fout:
        fout.write(f"Elapsed time: {time.time() - ss} seconds.")
