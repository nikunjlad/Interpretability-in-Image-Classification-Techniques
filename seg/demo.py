

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine.defaults import DefaultPredictor

cfg = get_cfg()
cfg.merge_from_file("configs/panoptic_fpn_R_50_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.WEIGHTS = "../models/model_final_c10459.pkl"
cfg.MODEL.DEVICE = "cpu"

img = "samples/biloxi.jpg"
predictor = DefaultPredictor(cfg)
