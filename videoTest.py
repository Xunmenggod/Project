import time
import cv2
import display
import numpy as np

def videoTest( model, class_names, videoSource):
    # capture video from camera, if videoSource = 0, open the camera
    cap = cv2.VideoCapture(videoSource)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 256)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 256)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    size = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    )
    # video_downsample_factor = 10
    codec = cv2.VideoWriter_fourcc(*'DIVX')
    video_outout = cv2.VideoWriter('detection_video.avi', codec, 5, size)
    # frame_counter = 0
    i = 0
    counter = 0
    images = {}
    checkPoint = 50

    while(cap.isOpened()):
        if i == 0:
            start_time = time.time()
        if i == checkPoint:
            frame_rate = display.calFrameRate(start_time, time.time(), checkPoint)
            print("frame rate:", frame_rate)

        ret, image = cap.read()

        i = i + 1
        if ret:
            #testing
            # cv2.imshow('Frame', images[i])
            filename = 'saveImage'

            #model detection
            result = model.detect([image], verbose=0)
            r = result[0]
            action = decision(r['rois'], 1020, 1080)
            print("decision for the user next action: ", action)
            instance = display.displayInstance(image, r['rois'], r['masks'],
                                                   r['class_ids'], class_names, r['scores'])
            # x = instance.shape[0]
            # y = instance.shape[1]
            # instance = cv2.resize(instance, (size[0] * 4, size[1] * 4))
            video_outout.write(instance)
            cv2.imshow('Frame', instance)

            if cv2.waitKey(1) & 0xff == ord('q'): # for quit
                break
            elif cv2.waitKey(5) & 0xff == ord(' '): # for pause
                cv2.waitKey(100)
                while True:
                    if cv2.waitKey(10) & 0xff == ord('s'): # for save
                        cv2.imwrite('./save/' + filename + str(counter) +'.jpg', instance)
                        print("save", counter + 1,"th image successfully")
                        counter = counter + 1
                    elif cv2.waitKey(10) & 0xff == ord(' '): # for cancel pause
                        cv2.waitKey(100)
                        break
            elif cv2.waitKey(10) & 0xff == ord('s'):  # for save
                cv2.imwrite('./save/' + filename + str(counter) + '.jpg', instance)
                print("save", counter + 1, "th image successfully")
                counter = counter + 1

    cap.release()
    cv2.destroyAllWindows()

# decision API
from enum import Enum
class Report(Enum):
    STOP = 0
    GO = 1  # go straight forward
    LEFT = 2
    RIGHT = 3


def decision(boxes, width, height, maxNumChecking = 5, threshold = 0.7):
    #inputs:
    # width: image.shape[1]
    # height: image.shape[0]
    # threshold: decision factor: range:[0, 1]
    # outputs:
    # one command of the report

    res = Report.GO
    # logic decision with bounding box coordinate and mask value
    width_threshold = width / 3
    height_threshold = height / 3
    image_area = width * height
    area_threshold = threshold * image_area / 9
    middle_boxes = []
    left_boxes = []
    right_boxes = []
    N = 0

    for i, box in enumerate(boxes):
        y1, x1, y2, x2 = box
        # extract center
        x = x2 - x1
        y = y2 - y1
        # boxes extraction
        # if x > height - height_threshold and x < height:
        if y > width_threshold and y < width - width_threshold:
            middle_boxes.append(box)
            N += 1
        # left box extraction
        if y > 0 and y < width_threshold:
            left_boxes.append(box)
        # right box extraction
        if y > width - width_threshold and y < width:
            right_boxes.append(box)


    # checking area of some random middle boxes
    checkingTime = 0
    if N > maxNumChecking:
        checkingTime = maxNumChecking
    else:
        checkingTime = N

    print("middle box number: ", N)
    randomIndex = np.random.randint(N, size=checkingTime)
    for i in range(checkingTime):
        m_y1, m_x1, m_y2, m_x2 = middle_boxes[randomIndex[i]]
        m_area = (m_y2 - m_y1) * (m_x2 - m_x1)
        if m_area > area_threshold:
            left_empty = 1
            right_empty = 1
            for j, left_box in enumerate(left_boxes):
                l_y1, l_x1, l_y2, l_x2 = left_box
                l_area = (l_y2 - l_y1) * (l_x2 - l_x1)
                if l_area > area_threshold:
                    left_empty = 0
                    break
                else:
                    left_empty = 1
            for j, right_box in enumerate(right_boxes):
                r_y1, r_x1, r_y2, r_x2 = right_box
                r_area = (r_y2 - r_y1) * (r_x2 - r_x1)
                if r_area > area_threshold:
                    right_empty = 0
                    break
                else:
                    right_empty = 1

            if left_empty == 0 and right_empty == 0:
                res = Report.STOP
                break
            elif left_empty == 0 and right_empty == 1:
                res = Report.RIGHT
            elif left_empty == 1 and right_empty == 0:
                res = Report.LEFT

        else:
            res = Report.GO

    return res

if __name__ == '__main__':
    import os
    import sys
    import random
    import math

    ROOT_DIR = os.path.abspath("../")
    sys.path.append(ROOT_DIR)  # To find local version of the library
    from mrcnn import utils
    import mrcnn.model as modellib

    sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
    import coco

    MODEL_DIR = os.path.join(ROOT_DIR, "logs")
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
    if not os.path.exists(COCO_MODEL_PATH):
        print("wrong path of directories")


    class InferenceConfig(coco.CocoConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1


    config = InferenceConfig()
    config.display()

    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

    # Load weights trained on MS-COCO
    model.load_weights(COCO_MODEL_PATH, by_name=True)

    # add up for other road
    class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
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
                   'teddy bear', 'hair drier', 'toothbrush']
    # video_source = 'test.mp4'
    video_source = 0
    videoTest(model=model, class_names=class_names, videoSource=video_source)
