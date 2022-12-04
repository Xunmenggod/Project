import cv2
import numpy as np
import time

# visualize by opencv
def randomColors(N):
    np.random.seed(1)
    colors = [tuple(255 * np.random.rand(3)) for _ in range(N)]
    return colors

def applyMask(image, mask, color, alpha=0.5):
    for n, c in enumerate(color):
        image[:, :, n] = np.where(mask == 1,
                                  image[:, :, n] *
                                  (1 - alpha) + alpha * c,
                                  image[:, :, n])
    return image

def displayInstance(image, boxes, masks, class_ids, class_names, scores):
    N = boxes.shape[0]
    if not N:
        print("no processing instance to display")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    colors = randomColors(N)
    height, weight = image.shape[:2]

    for i, color in enumerate(colors):
        if not np.any(boxes[i]):
            continue
        y1, x1, y2, x2 = boxes[i]
        mask = masks[:, :, i]
        label = class_names[class_ids[i]]
        score = scores[i] if scores is not None else None
        if score < 0.7:
            continue

        image = applyMask(image, mask, color)
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        caption = '{} {:.2f}'.format(label, score) if score else label
        image = cv2.putText(image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2)

    return image

def calFrameRate(start, end, numOfProcess):
    time_interval = (end - start) / numOfProcess * 1000 % 1000
    frame_rate = 1 / (time_interval * 0.001)
    return frame_rate

# def train(): # need to be implemented for class expansion


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

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    counter = 0
    while True:
        # time counting
        numOfProcess = 50
        if not counter:
            start_time = time.time()

        if counter == numOfProcess:
            frame_rate = calFrameRate(start_time, time.time(), numOfProcess)
            print("average frame rate: ", frame_rate)
            counter = 0

        ret, frame = cap.read()
        if ret:
            result = model.detect([frame], verbose=0)
            r = result[0]
            frame = displayInstance(frame, r['rois'], r['masks'],
                                    r['class_ids'], class_names, r['scores'])
            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xff == ord('q'):
                break

        counter = counter + 1

    cap.release()
    cap.destroyAllWindows()
