######################################################
##############INFERENCE###############################


predictor = DefaultPredictor(cfg)
from detectron2.utils.visualizer import ColorMode
import random
import matplotlib.pyplot as plt

object_metadata = MetadataCatalog.get("object_val")
cfg.DATASETS.TEST = ("object_val", )
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a testing threshold
predictor = DefaultPredictor(cfg)

from detectron2.utils.visualizer import ColorMode
    
im = cv2.imread("G:/detectron2/val/aug_96_344.jpeg")
outputs = predictor(im) 

v = Visualizer(im[:, :, ::-1],
                   metadata=object_metadata, 
                   scale=0.4                
    )

out = v.draw_instance_predictions(outputs["instances"].to("cpu")) #for single image CPU is fine
plt.figure(figsize=(720,540))
plt.imshow(out.get_image()[:, :, ::-1][..., ::-1])