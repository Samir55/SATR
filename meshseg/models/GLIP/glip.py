import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo


class GLIPModel: 
    def __init__(self):
        # ! wget https://penzhanwu2bbs.blob.core.windows.net/data/GLIPv1_Open/models/glip_large_model.pth -O /media/abdeas0a/Ahmed/research/GLIP/MODEL/glip_large_model.pth
        config_file = (
            "GLIP/configs/pretrain/glip_Swin_L.yaml"
        )
        weight_file = "GLIP/MODEL/glip_large_model.pth"

        # update the config options with the config file
        # manual override some options
        cfg.local_rank = 0
        cfg.num_gpus = 1
        cfg.merge_from_file(config_file)
        cfg.merge_from_list(["MODEL.WEIGHT", weight_file])
        cfg.merge_from_list(["MODEL.DEVICE", "cuda"])
        
        self.model = GLIPDemo(
            cfg, min_image_size=800, confidence_threshold=0.7, show_mask_heatmaps=False
        )

    def predict(self, img, text):
        with torch.no_grad():
            return self.model.run_on_web_image(img, text, 0.5)
