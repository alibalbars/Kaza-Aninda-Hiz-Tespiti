saniye = ((0 * 60) + 10) * 30 + 0
isim = "normal_traffic1.mp4"

from absl import flags
import sys
FLAGS = flags.FLAGS
FLAGS(sys.argv)

import time
import numpy as np
import cv2
import matplotlib.pyplot as plt

import tensorflow as tf
from yolov3_tf2.models import YoloV3
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import convert_boxes

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

from MY_print_methods import print_class
from RAGHAV_object_tracker import object_tracker

from _collections import deque

p = print_class()
ot = object_tracker()
is_init_frame = True # Flag is necessary to setup object tracking properly
prev_frame_objects = []
cur_frame_objects = []
font = cv2.FONT_HERSHEY_SIMPLEX # OpenCV font for drawing text on frame

crash_flag = False


class_names = [c.strip() for c in open('./data/labels/coco.names').readlines()]
yolo = YoloV3(classes=len(class_names))
yolo.load_weights('./weights/yolov3.tf')

max_cosine_distance = 0.5
nn_budget = None
nms_max_overlap = 0.8

model_filename = 'model_data/mars-small128.pb'
encoder = gdet.create_box_encoder(model_filename, batch_size=1)
metric = nn_matching.NearestNeighborDistanceMetric('cosine', max_cosine_distance, nn_budget)
tracker = Tracker(metric)

vid = cv2.VideoCapture('./data/video/' + isim) # 28, 26, 30 (mTracker güzel test)
vid.set(1, saniye)


# İlk Frame al

# Obje Tracker başlat
# mTracker = cv2.TrackerMOSSE_create()
# _, img = vid.read()
# tbox = cv2.selectROI("Tracking", img, False)
# mTracker.init(img, tbox)


# codec = cv2.VideoWriter_fourcc(*'XVID')
# vid_fps =int(vid.get(cv2.CAP_PROP_FPS))
# vid_width,vid_height = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
# out = cv2.VideoWriter('./data/video/results.avi', codec, vid_fps, (vid_width, vid_height))

# pts = [deque(maxlen=30) for _ in range(1000)]

while True:
    _, img = vid.read()
    if img is None:
        print('Completed')
        break

    # tbox yenile
    #success, tbox = mTracker.update(img)
    # tbox çiz
    #cv2.rectangle(img, (int(tbox[0]),int(tbox[1])), (int(tbox[0]) + int(tbox[2]), int(tbox[1]) + int(tbox[3])), (204, 235, 52), 2)


    #draw_frame = img.copy()

    img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_in = tf.expand_dims(img_in, 0)
    img_in = transform_images(img_in, 416)

    # print("-------------------------------")
    # print(type(img_in)) # <class 'tensorflow.python.framework.ops.EagerTensor'>
    # print(img_in.shape) # (1, 416, 416, 3)
    # print(img_in.dtype) # <dtype: 'float32'>
    # print("-------------------------------")

    t1 = time.time()

    boxes, scores, classes, nums = yolo.predict(img_in)

    classes = classes[0]
    names = []
    for i in range(len(classes)):
        names.append(class_names[int(classes[i])])

    names = np.array(names)

    converted_boxes = convert_boxes(img, boxes[0])

    features = encoder(img, converted_boxes)

    detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
                  zip(converted_boxes, scores[0], names, features)]

    boxs = np.array([d.tlwh for d in detections])
    scores = np.array([d.confidence for d in detections])
    classes = np.array([d.class_name for d in detections])
    indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
    detections = [detections[i] for i in indices]
    #p.print(detections)  // [<deep_sort.detection.Detection object at 0x00000218F72D8D68>, <deep_sort.detection.Detection object at 0x00000218F7325940>, <deep_sort.detection.Detection object at 0x00000218F7325C18>, <deep_sort.detection.Detection object at 0x00000218F7325F28>, <deep_sort.detection.Detection object at 0x00000218F7325EF0>, <deep_sort.detection.Detection object at 0x00000218F7325F60>, <deep_sort.detection.Detection object at 0x00000218F7325D68>, <deep_sort.detection.Detection object at 0x00000218F7325DA0>, <deep_sort.detection.Detection object at 0x00000218F7329400>]
    #p.type_(detections) // TYPE => <class 'list'>
    # p.print(detections[1]) // <deep_sort.detection.Detection object at 0x00000266814E3828>

    tracker.predict()
    tracker.update(detections)

    # Matplotlib has a number of built-in colormaps accessible via matplotlib.cm.get_cmap.
    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i)[:3] for i in np.linspace(0,1,20)]

    # Current vehicle count
    current_count = int(0)

    # tracker'ın tüm sonuçları için for döngüsü
    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update >1:
            continue
        bbox = track.to_tlbr() # [848.78925062 113.98058018 901.1299524  144.32627563]
        class_name = track.get_class() # car (nesne ismi)
        color = colors[int(track.track_id) % len(colors)] # (0.807843137254902, 0.8588235294117647, 0.611764705882353)

        color = [i * 255 for i in color] # [231.0, 203.0, 148.0]
        p.print("GELDİ")

        # img => videodan alınan frame (np ndarray)

        #Bounding box çiz
        # cv2.rectangle(img, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])), color, 2)
        # #cv2.rectangle(img, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)
        #             #+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
        # cv2.putText(img, class_name+"-"+str(track.track_id), (int(bbox[0]), int(bbox[1]-10)), 0, 0.75,
        #             (255, 255, 255), 2)

        # Merkez bul
        center = (int(((bbox[0]) + (bbox[2]))/2), int(((bbox[1])+(bbox[3]))/2)) # (665, 113),  class = tuple
        # center = (int(((tbox[0]) + (tbox[2]/2.0))), int(((tbox[1])+(tbox[3]/2.0))))
        # track_id => sürekli artıyor sınırsız şekilde, yavaş yavaş artıyor

        # Yeni kod
        if (is_init_frame == True):
            # prev_frame_objects.append([(center[0], center[1]), ot.get_init_index(), 0, deque(), -1, 0, bbox])
            prev_frame_objects.append([(center[0], center[1]), ot.find_next_free_index(), 0, deque(), -1, 0, bbox])
        else:
            cur_frame_objects.append([(center[0], center[1]), 0, 0, deque(), -1, 0, bbox])

    if (is_init_frame == False):
        # We only run when we have had at least 1 object detected in the previous (initial) frame
        if (len(prev_frame_objects) != 0):
            cur_frame_objects = ot.sort_cur_objects(prev_frame_objects, cur_frame_objects)
            p.print(cur_frame_objects)


    # FPS hesapla
    fps = 1./(time.time()-t1)
    cv2.putText(img, "FPS: {:.2f}".format(fps), (0,30), 0, 1, (0,0,255), 2)
    cv2.resizeWindow('output', 1024, 768)
    # cv2.imshow('output', img) #####################################
    # out.write(img)

    is_crash_detected = False # Has a crash been detected anywhere in our current frame?
    for point in cur_frame_objects: # Iterating through all our objects in the current frame.
        # Only objects that have been present for 5 consecutive frames are considered. This is done to
        # filter out any inaccurate momentary detections.
        if (point[2] >= 5):
            # Finding vector of object across 5 frames
            vector = [point[3][-1][0] - point[3][0][0], point[3][-1][1] - point[3][0][1]]
            # Getting a simple estimate coordinate of where we expect our object to end up
            # with its current vector. This is used to draw the predicted vector for each object.
            end_point = (2 * vector[0] + point[3][-1][0], 2 * vector[1] + point[3][-1][1])

            # Getting magnitude of vector for crash detection. We could use the direction in this detection
            # as well, but we achieved much better results when just using the magnitude.
            vector_mag = (vector[0]**2 + vector[1]**2)**(1/2)
            # Change in magnitude (essentially the object's acceleration/deceleration)
            delta = abs(vector_mag - point[5])

            # Flag for current object being a crash or not.
            has_object_crashed = False
            if (delta >= 11) and point[5] != 0.0: # Criteria for crash.
                is_crash_detected = True
                has_object_crashed = True

            # Drawing circle to label detected objects
            cv2.circle(img, point[0], 5, (0, 255, 255), 2) #draw_frame
            if (has_object_crashed == True):
                # Red circle is drawn around an object that has been suspected of crashing
                cv2.circle(img, point[0], 40, (0, 0, 255), 4) #draw_frame

            # Drawing predicted future vector of each object. (Blue line)
            # Vektör çiz
            cv2.line(img, point[3][-1], end_point, (255, 255, 0), 2) #draw_frame
            cv2.putText(img, str(point[1]), (point[0][0], point[0][1] + 30), font, 1, (255, 50, 50), 2, cv2.LINE_AA)

    if (is_crash_detected == True):
        #cv2.putText(img, f"CRASH DETECTED", (0, 30), font, 1, (0, 0, 255), 2, cv2.LINE_AA) #draw_frame
        p.print(" KAZA OLDUUUUUUUUUUUUUUUUUUUUUUUUUUU ")
        crash_flag = True
        cv2.putText(img, f"CRASH DETECTED", (300, 300), font, 1, (252, 102, 3), 2, cv2.LINE_AA) #draw_frame

    if crash_flag == True:
        cv2.putText(img, f"CRASH DETECTED", (300, 30), font, 1, (0, 0, 255), 2, cv2.LINE_AA) #draw_frame



    if (is_init_frame == False):
        prev_frame_objects = cur_frame_objects.copy()
        cur_frame_objects = []
    is_init_frame = False

    cv2.imshow('Tracking', img)

    if cv2.waitKey(1) == ord('q'):
        break
vid.release()
# out.release()
cv2.destroyAllWindows()
