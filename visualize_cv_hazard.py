import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import cv2
import time
import imutils
import threading

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
#from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")
VIDEO_DIR = os.path.join(ROOT_DIR, "videos")

CAMERA_NUMBER = 1

HAZARD = False
HAZARD_LOC = 'S'
##############################################################
class_hazards = [
    	'backpack', 'handbag', 'tie', 'frisbee', 'sports ball', 'tennis racket','bottle','keyboard','book', 'fire hydrant', 'bird'
    	]
###############################################################

def random_colors(N):
    np.random.seed(1)
    colors = [tuple(255 * np.random.rand(3)) for _ in range(N)]
    return colors

def filer_object(label):
    filtered_class_names = [
        'airplane', 'boat', 'cat', 'dog', 'horse', 'sheep', 'cow', 
        'elephant', 'bear', 'zebra', 'giraffe', 'umbrella',
        'suitcase', 'skis', 'snowboard', 
        'surfboard', 'tennis racket', 
        'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
        'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
        'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
        'sink', 'refrigerator',  'clock', 'vase', 'scissors',
        'teddy bear', 'hair drier', 'toothbrush'
    ]
    if(label in filtered_class_names):
        return True
    
    return False


def apply_mask(image, mask, color, alpha=0.5):
    """apply mask to image"""
    for n, c in enumerate(color):
        image[:, :, n] = np.where(mask == 1, image[:, :, n] * (1 - alpha) + alpha * c , image[:, :, n])
    return image

def display_instances(image, boxes, masks, ids, names, scores):
    """
        take the image and results and apply the mask, box, and Label
    """
    global class_hazards, HAZARD, HAZARD_LOC
    n_instances = boxes.shape[0]
    colors = random_colors(n_instances)

    if not n_instances:
        print('NO INSTANCES TO DISPLAY')
    else:
        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]

    box_image = image.copy()
    masked_image = np.zeros(image.shape)

    for i, color in enumerate(colors):
        if not np.any(boxes[i]):
            continue

        y1, x1, y2, x2 = boxes[i]
        label = names[ids[i]]

        start = time.time() *1000.0
        if(filer_object(label)):
            continue
        ######################################
        if (label in class_hazards):
            HAZARD = True
            label = 'Hazard'
            center_x = (int(x1)+ int(x2))/2.0
            #center_y = (y1+y2)/2.0
            if(center_x>image.shape[1]/2):
                print ('Hazard in right')
                HAZARD_LOC = 'R'
            elif(center_x<image.shape[1]/2):
                HAZARD_LOC = 'L'
                print ('Hazard in left')

        ######################################
        end = time.time() *1000.0
        print ('Application output time : ' , (end-start) , ' ms')

        score = scores[i] if scores is not None else None
        caption = '{} {:.2f}'.format(label, score) if score else label
        mask = masks[:, :, i]

        masked_image = apply_mask(masked_image, mask, color)
        box_image = cv2.rectangle(box_image, (x1, y1), (x2, y2), color, 2)
        image = cv2.putText(
            box_image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2
        )
    return masked_image, box_image

###############################################
h_frame = np.zeros((480,800,3))
font = cv2.FONT_HERSHEY_SIMPLEX
h_text = "Hazard Ahead!"
advisory_text = "Change Direction!"

textSize = cv2.getTextSize(h_text,font,3,2)[0]
textX = int((h_frame.shape[1] - textSize[0])/2)
textY = int((h_frame.shape[0] - textSize[1])/2) + 20
###############################################

PROCESS_DONE = False
ori_frame = np.zeros((800,480,3)) 
frame = np.zeros((800,480,3))
m_frame = np.zeros((800,480,3))

class HazardVisualization(threading.Thread):
    #def __init__(self):
    #    Thread.__init__(self)
    #    print ('[+] New Thread for Hazard visualize created!')

    def run(self):
        print ('[+] New Thread for Hazard visualize created!')
        global ori_frame, m_framem, frame, PROCESS_DONE, HAZARD

        while  True:
            if(PROCESS_DONE == True):
                #if(ori_frame!= None and frame!=None and m_frame!=None):
                cv2.imshow('[+] Original Frame', ori_frame)
                cv2.imshow('[+] Object Detection', frame)
                cv2.imshow('[+] Semantic Segmentation', m_frame)

                h_frame[:] = (0,0,255)
                if(HAZARD == True):
                    h_frame = cv2.putText(h_frame, h_text, (textX, textY-50), font, 3 , (0,0,0), 2)
                    h_frame = cv2.putText(h_frame, advisory_text, (textX+50, textY+50), font, 2 , (0,0,0), 2)
            
                cv2.imshow('Warning',h_frame)
            time.sleep(0.1)
#############################################################
if __name__ == '__main__':
    """
        test everything
    """
    #import os
    #import sys
    #import coco
    #import utils
    #import model as modellib

    #ROOT_DIR = os.getcwd()
    #MODEL_DIR = os.path.join(ROOT_DIR, "logs")
    #COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
    #if not os.path.exists(COCO_MODEL_PATH):
    #    utils.download_trained_weights(COCO_MODEL_PATH)

    class InferenceConfig(coco.CocoConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    config = InferenceConfig()
    config.display()

    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
    model.load_weights(COCO_MODEL_PATH, by_name=True)
    
    class_names = [
        'BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
        'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
        'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
        'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
        'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard',
        'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
        'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
        'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
        'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
        'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
        'teddy bear', 'hair drier', 'toothbrush'
    ]

    
    capture = cv2.VideoCapture(CAMERA_NUMBER)


    # these 2 lines can be removed if you dont have a 1080p camera.
    #capture.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
    #capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 400)

    print('Now capturing camera data!')
    tasks = [HazardVisualization()]

    #for t in tasks:
    #    t.start()


    while True:
        ret, frame = capture.read()
        frame = imutils.rotate(frame, 180)
        #img = frame.copy()
        #frame = cv2.flip(img,1)
        ori_frame = frame.copy()
        #frame = cv2.resize(frame,(600,400))
        
        PROCESS_DONE = False

        start = time.time()
        results = model.detect([frame], verbose=0)
        end = time.time()
        r = results[0]
        HAZARD =  False
        
        m_frame , frame = display_instances(
            frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores']
        )
        
        ori_frame = cv2.resize(ori_frame,(800,480))
        frame = cv2.resize(frame,(800,480))
        m_frame = cv2.resize(m_frame,(800,480))

        PROCESS_DONE = True

        #cv2.imshow('[+] Original Frame', ori_frame)
        #cv2.imshow('[+] Object Detection', frame)
        #cv2.imshow('[+] Semantic Segmentation', m_frame)

        h_frame[:] = (0,0,255)
        if(HAZARD == True):
            h_frame = cv2.putText(h_frame, h_text, (textX, textY-50), font, 3 , (0,0,0), 2)
            #h_frame = cv2.putText(h_frame, advisory_text, (textX+50, textY+50), font, 2 , (0,0,0), 2)
            
            if(HAZARD_LOC == 'L'):
                advisory_text = "Steer to right"
                print('Go Right!')
            if (HAZARD_LOC =='R'):
                print('Go Left')
                advisory_text = "Steer to left"
            h_frame = cv2.putText(h_frame, advisory_text, (textX+50, textY+50), font, 2 , (0,0,0), 2)
            
    
        #cv2.imshow('Warning',h_frame)        
        
        cv2.imshow('Original Frame',ori_frame)
        cv2.imshow('Object Detection', frame)
        cv2.imshow('Segmentation',m_frame)
        cv2.imshow('Warning',h_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        print ('Inference time : ' , end-start)
    capture.release()
cv2.destroyAllWindows()