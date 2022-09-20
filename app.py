import cv2
import sys
import tkinter
from tkinter import Frame, Tk, BOTH, Text, Menu, END
from tkinter.filedialog import Open, SaveAs
import time
import numpy as np
import os.path
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import joblib
import torch
assert torch.__version__.startswith("1.8") 
import torchvision
import cv2
import os
import numpy as np
import json
import random
import matplotlib.pyplot as plt
import tqdm
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
def get_data_dicts(directory, classes):
    dataset_dicts = []
    for filename in [file for file in os.listdir(directory) if file.endswith('.json')]:
        json_file = os.path.join(directory, filename)
        with open(json_file) as f:
            img_anns = json.load(f)

        record = {}
        
        filename = os.path.join(directory, img_anns["imagePath"])
        
        record["file_name"] = filename
        record["height"] = 480
        record["width"] = 480
    
        annos = img_anns["shapes"]
        objs = []
        for anno in annos:
            px = [a[0] for a in anno['points']] # x coord
            py = [a[1] for a in anno['points']] # y-coord
            poly = [(x, y) for x, y in zip(px, py)] # poly for segmentation
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": classes.index(anno['label']),
                "iscrowd": 0
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

classes = ['Xoai']

data_path = 'data2/'

for d in ["train", "test"]:
    DatasetCatalog.register(
        "category_" + d, 
        lambda d=d: get_data_dicts(data_path+d, classes)
    )
    MetadataCatalog.get("category_" + d).set(thing_classes=classes)

microcontroller_metadata = MetadataCatalog.get("category_train")
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode, Visualizer
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("category_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 1000
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
# trainer = DefaultTrainer(cfg) 
# trainer.resume_or_load(resume=False)
#trainer.train()
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 
cfg.DATASETS.TEST = ("Xoai", )
predictor = DefaultPredictor(cfg)

# img = cv2.imread(x)
# outputs = predictor(img)
# v = Visualizer(img[:, :, ::-1],metadata=microcontroller_metadata, scale=1, instance_mode=ColorMode.IMAGE_BW # removes the colors of unsegmented pixels
#     )
# a = outputs["instances"].pred_boxes.tensor.cpu().numpy()
# a.astype(int)
# x = str(np.round(a[0,2]-a[0,0]))
# y = str(np.round(a[0,3]-a[0,1]))

# text = "Kich thuoc:"+x+" x "+y
# print(text)
# for box in outputs["instances"].pred_boxes.to('cpu'):
#     v.draw_box(box)
#     b = outputs["instances"].scores.to('cpu').numpy()
#     score = np.round(b[0]*100, 2)
#     score = "Xoài: "+ str(score)+"%"
#     v.draw_text(score, tuple(box[:2].numpy()),font_size=15,color='r', horizontal_alignment='left', rotation=0)
# #out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
# out = v.draw_text(text,(10,10),font_size=15,color='g', horizontal_alignment='left', rotation=0)

def load_image(path):
    img = cv2.imread(path, 1)
    # OpenCV loads images with color channels
    # in BGR order. So we need to reverse them
    return img[...,::-1]
class Main(Frame):
    
    def __init__(self, parent):
        Frame.__init__(self, parent)
        self.parent = parent
        self.initUI()
  
    def initUI(self):
        self.parent.title("Nhan Dang Xoai")
        self.pack(fill=BOTH, expand=1)
  
        menubar = Menu(self.parent)
        self.parent.config(menu=menubar)
  
        fileMenu = Menu(menubar)
        fileMenu.add_command(label="OpenImg", command=self.onOpenImg)
        fileMenu.add_command(label="OpenVideo", command=self.onOpenVideo)
        fileMenu.add_command(label="RecognitionImg", command=self.onRecognition)
        fileMenu.add_command(label="RecognitionVideo", command=self.onRecognition1)
        fileMenu.add_separator()
        fileMenu.add_command(label="Exit", command=self.quit)
        menubar.add_cascade(label="File", menu=fileMenu)
        self.txt = Text(self)
        self.txt.pack(fill=BOTH, expand=1)
  
    def onOpenImg(self):
        global ftypes
        ftypes = [('Images', '*.jpg *.tif *.bmp *.gif *.png')]
        dlg = Open(self, filetypes = ftypes)
        fl = dlg.show()
  
        if fl != '':
            global img
            global imgin
            imgin = cv2.imread(fl)
            cv2.namedWindow("ImageIn", cv2.WINDOW_AUTOSIZE)
            #cv2.moveWindow("ImageIn", 200, 200)
            cv2.imshow("ImageIn", imgin)
    def onOpenVideo(self):
        global ftypes
        ftypes = [('Videos', '*.mp4')]
        dlg = Open(self, filetypes = ftypes)
        fl = dlg.show()
  
        if fl != '':
            global img
            global video
            video = cv2.VideoCapture(fl)
            time.sleep(1)
            if video is None or not video.isOpened():
                print('Khong the mo file video')
                return
        # width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        # height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # frames_per_second = video.get(cv2.CAP_PROP_FPS)
        # num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        # Initialize video writer
        # video_writer = cv2.VideoWriter('out.mp4', fourcc=cv2.VideoWriter_fourcc(*"mp4v"), fps=float(frames_per_second), frameSize=(width, height), isColor=True)

        # Initialize predictor

        # Initialize visualizer
        #v = VideoVisualizer(MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), ColorMode.IMAGE)
        
        # v = VideoVisualizer(MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), ColorMode.IMAGE)

        # def runOnVideo(video, maxFrames):
        #     """ Runs the predictor on every frame in the video (unless maxFrames is given),
        #     and returns the frame with the predictions drawn.
        #     """

        #     readFrames = 0
        #     while True:
        #         hasFrame, frame = video.read()
        #         ch = cv2.waitKey(30)
        #         if not hasFrame:
        #             break

        #         # Get prediction results for this frame
        #         outputs = predictor(frame)

        #         # Make sure the frame is colored
        #         frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        #         # Draw a visualization of the predictions using the video visualizer
        #         visualization = v.draw_instance_predictions(frame, outputs["instances"].to("cpu"))

        #         # Convert Matplotlib RGB format to OpenCV BGR format
        #         visualization = cv2.cvtColor(visualization.get_image(), cv2.COLOR_RGB2BGR)

        #         yield visualization

        #         readFrames += 1
        #         if readFrames > maxFrames:
        #             break

        # # Create a cut-off for debugging
        # num_frames = 20
        # cv2.namedWindow('Image2', cv2.WINDOW_AUTOSIZE)
        # # Enumerate the frames of the video
        # for visualization in tqdm.tqdm(runOnVideo(video, num_frames), total=num_frames):

        #     # # Write test image
        #     # cv2.namedWindow('Image2', cv2.WINDOW_AUTOSIZE)
        #     cv2.imshow('Image2', visualization)
            # cv2.imwrite('POSE detectron2.png', visualization)

            # # Write to video file
            # video_writer.write(visualization)

        # Release resources
        # video.release()
        # video_writer.release()
        # cv2.destroyAllWindows()
        
        # cv2.namedWindow("ImageIn", cv2.WINDOW_AUTOSIZE)
        #cv2.moveWindow("ImageIn", 200, 200)
        # cv2.imshow("ImageIn", imgin)
    def onRecognition(self):
        #img_test = align_image(img)
        # scale RGB values to interval [0,1]
        # img_test = (img_test / 255.).astype(np.float32)
        # # obtain embedding vector for image
        # embedded_test = nn4_small2_pretrained.predict(np.expand_dims(img_test, axis=0))[0]
        # test_prediction = svc.predict([embedded_test])
        # result = mydict[test_prediction[0]]
        # cv2.putText(imgin,result,(5,15),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255))
        # cv2.namedWindow("ImageIn", cv2.WINDOW_AUTOSIZE)
        # cv2.imshow("ImageIn", imgin)
        # plt.figure(figsize = (14, 10))
        # plt.imshow(img)
        # plt.show()
        outputs = predictor(imgin)
        v = Visualizer(imgin[:, :, ::-1],metadata=microcontroller_metadata, scale=1, instance_mode=ColorMode.IMAGE_BW # removes the colors of unsegmented pixels
            )
        a = outputs["instances"].pred_boxes.tensor.cpu().numpy()
        a.astype(int)
        x = str(np.round(a[0,2]-a[0,0]))
        y = str(np.round(a[0,3]-a[0,1]))
        dem = 0
        text = "Kich thuoc:"+x+" x "+y
        print(text)
        for box in outputs["instances"].pred_boxes.to('cpu'):
            v.draw_box(box)
            b = outputs["instances"].scores.to('cpu').numpy()
            score = np.round(b[0]*100, 2)
            score = "Xoài: "+ str(score)+"%"
            v.draw_text(score, tuple(box[:2].numpy()),font_size=15,color='r', horizontal_alignment='left', rotation=0)
            dem = dem+1
            print(dem)
        #out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        out = v.draw_text(text,(10,10),font_size=15,color='g', horizontal_alignment='left', rotation=0)
        out =cv2.cvtColor(out.get_image(), cv2.COLOR_BGR2RGB)
        # cv2.imshow("Input", out)
        #out = v.draw_box((a[0,0],a[0,1],a[0,2],a[0,3]), alpha = 0.5 , edge_color = 'g' , line_style = '-' )
        # plt.figure(figsize = (14, 10))
        # plt.imshow(cv2.cvtColor(out.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
        # plt.show()
        print()
        cv2.namedWindow("ImageOut", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("ImageOut",out)
    def onRecognition1(self):
        cv2.namedWindow('Image2', cv2.WINDOW_AUTOSIZE)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        video_writer = cv2.VideoWriter('out.mp4', fourcc=cv2.VideoWriter_fourcc(*"mp4v"), fps=float(frames_per_second), frameSize=(width, height), isColor=True)

        def runOnVideo(video, maxFrames):
            """ Runs the predictor on every frame in the video (unless maxFrames is given),
            and returns the frame with the predictions drawn.
            """

            readFrames = 0
            while True:
                hasFrame, frame = video.read()
                if not hasFrame:
                    break

                # Get prediction results for this frame
                #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                outputs = predictor(frame)
                v = Visualizer(frame[:, :, ::-1],metadata=microcontroller_metadata, scale=1, instance_mode=ColorMode.IMAGE_BW )
                a = outputs["instances"].pred_boxes.tensor.cpu().numpy()
                a.astype(int)
                x = str(np.round(a[0,2]-a[0,0]))
                y = str(np.round(a[0,3]-a[0,1]))

                text = "Kich thuoc:"+x+" x "+y
                # Make sure the frame is colored

                # Draw a visualization of the predictions using the video visualizer
                for box in outputs["instances"].pred_boxes.to('cpu'):
                    v.draw_box(box)
                    b = outputs["instances"].scores.to('cpu').numpy()
                    score = np.round(b[0]*100, 2)
                    score = "Xoài: "+ str(score)+"%"
                    v.draw_text(score, tuple(box[:2].numpy()),font_size=15,color='r', horizontal_alignment='left', rotation=0)
                #visualization = v.draw_box((a[0,0],a[0,1],a[0,2],a[0,3]), alpha = 0.5 , edge_color = 'g' , line_style = '-' )
                visualization = v.draw_text(text,(10,10),font_size=15,color='g', horizontal_alignment='left', rotation=0)

                # Convert Matplotlib RGB format to OpenCV BGR format
                visualization = cv2.cvtColor(visualization.get_image(), cv2.COLOR_RGB2BGR)
                #visualization = cv2.resize(visualization,(width,height))

                yield visualization

                readFrames += 1
                if readFrames > maxFrames:
                    break

        # Create a cut-off for debugging
        num_frames = 120

        # Enumerate the frames of the video
        for visualization in tqdm.tqdm(runOnVideo(video, num_frames), total=num_frames):

            # Write test image
            cv2.imwrite('POSE detectron2.png', visualization)

            # Write to video file
            video_writer.write(visualization)


root = Tk()
Main(root)
root.geometry("480x480+100+100")
root.mainloop()