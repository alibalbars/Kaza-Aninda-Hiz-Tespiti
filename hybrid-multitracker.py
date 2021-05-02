#süre
# saniye = ((0 * 60) + 3) * 30 + 0
# name = "100"

saniye = ((5 * 60) + 48) * 30 + 20
name = "cctv1"
saniye = ((7 * 60) + 7) * 30 + 20
saniye = ((8 * 60) + 27) * 30 + 0

from absl import flags
import sys
import os

FLAGS = flags.FLAGS
FLAGS(sys.argv)

import time
import numpy as np
import cv2
import matplotlib.pyplot as plt

import tensorflow as tf
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
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

trackerTypes = ['BOOSTING', 'MIL', 'KCF','TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']

def createTrackerByName(trackerType):
  # Create a tracker based on tracker name
    if trackerType == trackerTypes[0]:
        tracker = cv2.TrackerBoosting_create()
    elif trackerType == trackerTypes[1]:
        tracker = cv2.TrackerMIL_create()
    elif trackerType == trackerTypes[2]:
        tracker = cv2.TrackerKCF_create()
    elif trackerType == trackerTypes[3]:
        tracker = cv2.TrackerTLD_create()
    elif trackerType == trackerTypes[4]:
        tracker = cv2.TrackerMedianFlow_create()
    elif trackerType == trackerTypes[5]:
        tracker = cv2.TrackerGOTURN_create()
    elif trackerType == trackerTypes[6]:
        tracker = cv2.TrackerMOSSE_create()
    elif trackerType == trackerTypes[7]:
        tracker = cv2.TrackerCSRT_create()
    else:
        tracker = None
        print('Incorrect tracker name')
        print('Available trackers are:')
        for t in trackerTypes:
            print(t)

    return tracker

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

vid = cv2.VideoCapture('./data/video/' + name + '.mp4')
# get video dimensions
width = 0
height = 0
if vid.isOpened():
    width  = vid.get(3)  # float `width`
    height = vid.get(4)  # float `height`

multiTracker = cv2.MultiTracker_create()

def main():

    ot = object_tracker()
    is_init_frame = True # Flag is necessary to setup object tracking properly
    prev_frame_objects = []
    cur_frame_objects = []
    font = cv2.FONT_HERSHEY_SIMPLEX # OpenCV font for drawing text on frame

    crash_flag = False

    vid = cv2.VideoCapture('./data/video/' + name + '.mp4') # 28, 26, 30 (mTracker güzel test)
    vid.set(1, saniye)


    # İlk Frame al

    # tracker başlat
    # mTracker = cv2.TrackerMOSSE_create()
    # _, img = vid.read()
    # tbox = cv2.selectROI("Tracking", img, False)
    # mTracker.init(img, tbox)

    codec = cv2.VideoWriter_fourcc(*'XVID')
    vid_fps =int(vid.get(cv2.CAP_PROP_FPS))
    vid_width,vid_height = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter('./data/video/results.avi', codec, vid_fps, (vid_width, vid_height))

    #kuyruk deque'si
    pts = [deque(maxlen=300) for _ in range(1000)]
    pts_index = 0

    counter = 0
    isBboxesFilled = False # yolo'nun sounç verdiği ilk kareyi tespit etmek için kullandık.

    listX_prev = [-1] * 50
    listY_prev = [-1] * 50

    while True:
        t1 = time.time()
        counter = counter + 1
        print(counter)

        _, img = vid.read()
        if img is None:
            print('Completed')
            break


        # sadece bir kere çalışır
        if counter == 1 or isBboxesFilled == False:
            bboxes = get_bboxes(img) # yolo

            # p.print("bboxes\n" + str(bboxes))

            # List'in içi dolu
            if bboxes:
                isBboxesFilled = True

                # convertedBboxes = []
                # for bbox in bboxes:
                #     w = (bbox[2] - bbox[0])
                #     h = (bbox[3] - bbox[1])
                #     convertedBbox = [bbox[0], bbox[1], w, h]
                #     convertedBboxes.append(convertedBbox)

                convertedBboxes = expand_bboxes(tlbr_to_xywh(bboxes), .3)

                multiTracker_add(img, convertedBboxes)
            # multiTracker.clear()

        # 7 karede bir yolo çalıştır, takibi tekrar başlat
        # if counter % 7 == 0:
        #     multiTracker_reset()
        #     bboxes = get_bboxes(img) # yolo
        #     multiTracker_add(img, expand_bboxes(tlbr_to_xywh(bboxes), .3))

        # multitracker'ın takip ettiği araçların konumlarını yenile
        success, tboxes = multiTracker.update(img)

        # ardışık kareler aynıysa ikincisini kontrol etme
        # if center[0] == x_prev and center[1] == y_prev:
        #     continue
        # x_prev = center[0]
        # y_prev = center[1]

        isFrameSame = True
        for i, tbox in enumerate(tboxes):
            center = getTboxCenter(tbox)
            # en az bir aracın konumunu farklı tespit ederse
            if not (center[0] == int(listX_prev[i]) and center[1] == int(listY_prev[i])):
                isFrameSame = False
            listX_prev[i] = center[0]
            listY_prev[i] = center[1]


        if isFrameSame:
            counter -= 1
            p.print("GELDIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII")
            continue



        # tbox çiz
        for i, tbox in enumerate(tboxes): #tbox x1, y1, w, h
            # p1 = (int(tbox[0]), int(tbox[1]))
            # p2 = (int(tbox[0] + tbox[2]), int(tbox[1] + tbox[3]))
            center = getTboxCenter(tbox)

            # kuyruk çiz (kullanıyoruz), center noktasını pts dizisine ekle
            pts[pts_index].append(center)

            # pts.length kadar dön ve kırmızı nokta çiz (geçmiş karelerdeki noktaları da çizer)
            for j in range(1, len(pts[pts_index])):
                if pts[pts_index][j-1] is None or pts[pts_index][j] is None:
                    continue
                cv2.circle(img, pts[pts_index][j], 1, (0, 0, 255), 4)

            p1 = getTboxTopLeft(tbox)
            p2 = getTboxBottomRight(tbox)
            print(tbox[0], tbox[1], tbox[2], tbox[3])
            # cv2.rectangle(img, (int(tbox[0]),int(tbox[1])), (int(tbox[0]) + int(tbox[2]), int(tbox[1]) + int(tbox[3])), (204, 235, 52), 2, 1) # p1-solüst(x1, y1) p2-sağ alt(x2, y2)
            cv2.rectangle(img, p1, p2, (204, 235, 52), 2, 1) # p1-solüst(x1, y1) p2-sağ alt(x2, y2)




        # RAGHAV'A CENTER'LARI VER
        for tbox in tboxes:
            center = getTboxCenter(tbox)
            # track_id => sürekli artıyor sınırsız şekilde, yavaş yavaş artıyor

            # Tek frame'deki tüm araç center'larını depola
            if (is_init_frame == True): # ilk frame ise araçların center'larını cur yerine prev'in içine doldur.
                prev_frame_objects.append([(center[0], center[1]), ot.get_init_index(), 0, deque(), -1, 0])
            else:
                # her frame de içi tekrar dolduruluyor
                cur_frame_objects.append([(center[0], center[1]), 0, 0, deque(), -1, 0])


        # (her frame için bir kere döner)
        if (is_init_frame == False):
            # We only run when we have had at least 1 object detected in the previous (initial) frame

            # prev boş olursa hiç bir cur eşlenemeyeceği için böyle bi kontrol var
            if (len(prev_frame_objects) != 0):
                # cur'un center dışındaki tüm parametre verilerini doldurur
                cur_frame_objects = ot.sort_cur_objects(prev_frame_objects, cur_frame_objects)
                # p.print(cur_frame_objects)
                # for car in cur_frame_objects:
                #     cv2.putText(img, str(car[1]), car[0], 0, 1, (0,0,255), 2)


        # FPS hesapla
        fps = 1./(time.time()-t1)
        cv2.putText(img, "FPS: {:.2f}".format(fps), (0,30), 0, 1, (0,0,255), 2)
        cv2.resizeWindow('output', 1024, 768)
        # cv2.imshow('output', img) #####################################
        # out.write(img)

        #                     0             1  2    3       4  5
        # point => [(center[0], center[1]), 0, 0, deque(), -1, 0]
        # 0: center koordinatları
        # 1: id
        # 2: obje kaç frame'dir tespit ediliyor
        # 3: deque
        # 4: (eşlenme verisi) herhangi bir prev obj ile eşlenip eşlenmediği verisi (yada eşleşmiş olduğu obj'nin indexi olabilir)
        # 5: Magnitude of object (PREVIOUS FRAME)

        is_crash_detected = False # Has a crash been detected anywhere in our current frame?
        for i, point in enumerate(cur_frame_objects): # Iterating through all our objects in the current frame.
            # Only objects that have been present for 5 consecutive frames are considered. This is done to
            # filter out any inaccurate momentary detections.
            if (point[2] >= 5): # obje 5 kareden fazladır tespit ediilyorsa

                # Örnek deque => deque([(421, 293), (422, 293), (425, 296), (426, 296), (426, 297)])

                # point[3][-1][0] => objenin dequesisnin son elemanının x değeri
                # point[3][0][0] => objenin dequesisnin ilk elemanının x değeri

                # point[3][-1][1] => objenin dequesinin son elemanının y değeri
                # point[3][0][1] => objenin dequesinin ilk elemanının y değeri

                # vector => 5 karedeki x farkı, 5 karedeki y farkı
                # vector[0] => 5 karedeki x farkı
                # vector[1] => 5 karedeki y farkı

                # Finding vector of object across 5
                vector = [point[3][-1][0] - point[3][0][0], point[3][-1][1] - point[3][0][1]]

                # Getting a simple estimate coordinate of where we expect our object to end up
                # with its current vector. This is used to draw the predicted vector for each object.

                # end_point => x farkı * 2 + cur'un son x değeri, y farkı * 2 + cur'un son y değeri
                # mevcut konum + son 5 karedeki değişim
                end_point = (2 * vector[0] + point[3][-1][0], 2 * vector[1] + point[3][-1][1])

                # Getting magnitude of vector for crash detection. We could use the direction in this detection
                # as well, but we achieved much better results when just using the magnitude.

                # vector[0] = son 5 karedeki x farkı
                # vector[1] = son 5 karedeki y farkı
                # vector_mag = son 5 karedeki konum farkı (piksel bazlı)
                vector_mag = (vector[0]**2 + vector[1]**2)**(1/2)

                # Change in magnitude (essentially the object's acceleration/deceleration)
                # vector_mag => (mevcut kare, mecvut kare-5) yer değiştirme
                # point[5] => (previous kare, previous kare-5) yer değiştirme

                # delta = araç ivmesi
                delta = abs(vector_mag - point[5])

                #bbox = point[]
                cv2.putText(img, str("ivme: " + "{:.2f}".format(delta)), (int(tboxes[i][0]), int(tboxes[i][1] - 20)), font, 1, (235, 55, 55), 2, cv2.LINE_AA)


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

                # Vektör çizgisini çiz
                cv2.line(img, point[3][-1], end_point, (255, 255, 0), 2) #draw_frame

        if (is_crash_detected == True):
            #cv2.putText(img, f"CRASH DETECTED", (0, 30), font, 1, (0, 0, 255), 2, cv2.LINE_AA) #draw_frame
            p.print(" KAZA OLDUUUUUUUUUUUUUUUUUUUUUUUUUUU ")
            crash_flag = True
            cv2.putText(img, f"CRASH DETECTED", (300, 300), font, 1, (0, 0, 255), 2, cv2.LINE_AA) #draw_frame

        if crash_flag == True:
            cv2.putText(img, f"CRASH DETECTED", (300, 30), font, 1, (0, 0, 255), 2, cv2.LINE_AA) #draw_frame



        if (is_init_frame == False): # ilk frame değil ise
            prev_frame_objects = cur_frame_objects.copy() # prev = cur
            cur_frame_objects = [] # cur = []
        is_init_frame = False

        cv2.imshow('Tracking', img)

        # yavaş
        # time.sleep(0.8)

        # orta yavaş
        # time.sleep(0.5)

        # orta hızlı
        # time.sleep(0.3)

        # hızlı
        time.sleep(0.15)

        # orta hızlı
        # time.sleep(0.10)

        # çok hızlı
        # time.sleep(0.05)

        if cv2.waitKey(1) == ord('q'):
            break

    vid.release()
    out.release()
    cv2.destroyAllWindows()

def get_bboxes(img):
    # YOLO START
    img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_in = tf.expand_dims(img_in, 0)
    img_in = transform_images(img_in, 416)

    # print("-------------------------------")
    # print(type(img_in)) # <class 'tensorflow.python.framework.ops.EagerTensor'>
    # print(img_in.shape) # (1, 416, 416, 3)
    # print(img_in.dtype) # <dtype: 'float32'>
    # print("-------------------------------")


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

    # (tek frame içindeki tüm araçlar için döner) tracker'ın tüm sonuçları için for döngüsü
    bboxes = []
    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update >1:
            continue
        bbox = track.to_tlbr() # [848.78925062 113.98058018 901.1299524  144.32627563]
        # class_name = track.get_class() # car (nesne ismi)
        # color = colors[int(track.track_id) % len(colors)] # (0.807843137254902, 0.8588235294117647, 0.611764705882353)
        # color = [i * 255 for i in color] # [231.0, 203.0, 148.0]
        bboxes.append(bbox)

        # img => videodan alınan frame (np ndarray)

        #Bounding box çiz
        # cv2.rectangle(img, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])), color, 2)
        # #cv2.rectangle(img, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)
        #             #+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
        # cv2.putText(img, class_name+"-"+str(track.track_id), (int(bbox[0]), int(bbox[1]-10)), 0, 0.75,
        #             (255, 255, 255), 2)
    return bboxes

def multiTracker_add(img, bboxes):
    for bbox in bboxes:
        multiTracker.add(createTrackerByName("MOSSE"), img, tuple(bbox))

def multiTracker_remove():
    multiTracker.clear()

def multiTracker_reset():
    global multiTracker
    multiTracker = cv2.MultiTracker_create()

def tlbr_to_xywh(bboxes):
    convertedBboxes = []
    for bbox in bboxes:
        w = (bbox[2] - bbox[0])
        h = (bbox[3] - bbox[1])
        convertedBbox = [bbox[0], bbox[1], w, h]
        convertedBboxes.append(convertedBbox)
    return convertedBboxes

def expand_bboxes(bboxes, percentage): # x, y, w, h input alır. (0.25 percentage verince %25 büyür)
    convertedBboxes = []
    for bbox in bboxes:
        bbox[0] -= percentage / 2.0 * bbox[2]; # sol üst [x]
        bbox[1] -= percentage / 2.0 * bbox[3]; # sol üst [y]
        bbox[2] *= 1 + percentage; # w
        bbox[3] *= 1 + percentage; # h

        if bbox[0] < 0:
            bbox[0] = 0

        if bbox[1] < 0:
            bbox[1] = 0

        if bbox[0] + bbox[2] > width:
            bbox[2] = width - bbox[0]

        if bbox[1] + bbox[3] > height:
            bbox[3] = height - bbox[1]
        convertedBbox = [bbox[0], bbox[1], bbox[2], bbox[3]]
        convertedBboxes.append(convertedBbox)
    return convertedBboxes

def getTboxCenter(tbox): # x, y, w, h
    p1 = (int(tbox[0]), int(tbox[1])) # sol üst (x, y)
    p2 = (int(tbox[0] + tbox[2]), int(tbox[1] + tbox[3])) # sağ alt (x, y)
    # Merkez bul
    center = (int((p1[0] + p2[0])/2), int((p1[1] + p2[1])/2))
    return center

def getTboxTopLeft(tbox):
    p1 = (int(tbox[0]), int(tbox[1]))
    return p1

def getTboxBottomRight(tbox):
    p2 = (int(tbox[0] + tbox[2]), int(tbox[1] + tbox[3]))
    return p2


if __name__ == "__main__":
    main()
