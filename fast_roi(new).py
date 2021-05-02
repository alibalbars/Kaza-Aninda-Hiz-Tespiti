#süre
saniye = ((0 * 60) + 0) * 30 + 10
name = "2"

mod_var = 0
mod_flag = 0

import time, sys, cv2, matplotlib.pyplot as plt, numpy as np

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

# isim
vid = cv2.VideoCapture('./data/video/' + name + '.mp4') # 28, 26, 30 (opencv_tracker güzel test)

# vid.set(cv2.CAP_PROP_FPS, 1)
vid.set(1, saniye)

# tracker başlat
# opencv_tracker = cv2.TrackerMIL_create()
# opencv_tracker = cv2.TrackerBoosting_create()
# opencv_tracker = cv2.TrackerMOSSE_create()
# opencv_tracker = cv2.TrackerKCF_create()
# bir kere
_, img = vid.read()


# tbox güncellenmesi lazım (tbox => tuple)
tbox = cv2.selectROI("YENI FAST ROI", img, False)
opencv_tracker.init(img, tbox)
prev_img = img.copy()

is_center_same = False

x_prev = -1
y_prev = -1

#kuyruk deque'si
pts = [deque(maxlen=30) for _ in range(1000)]
pts_index = 0

while True:
    _, img = vid.read()
    if img is None:
        print('Completed')
        break

    # opencv_tracker_box yenile
    # img => yeni büyük frame
    # opencv_tracker.init(prev_img, tbox)
    success, tbox = opencv_tracker.update(img)
    # prev_img = img.copy()

    # tbox'ın içindeki image'ı almamız lazım

    # 1 =>

    # opencv_tracker_box çiz
    cv2.rectangle(img, (int(tbox[0]), int(tbox[1])), (int(tbox[0]) + int(tbox[2]), int(tbox[1]) + int(tbox[3])), (204, 235, 52), 2)
    center = (int(((tbox[0]) + (tbox[2]/2.0))), int(((tbox[1])+(tbox[3]/2.0))))

    # center'ın x ve y noktalarını yazdır
    p.print(str(center[0]) + " || " +  str(center[1]))

    # seçili selectROI kaybolursa
    if center[0] == 0 and center[1] == 0:
        cv2.putText(img, f"KAYBOLDU", (0, 60), font, 1, (255, 50, 50), 2, cv2.LINE_AA)
        p.print("SHUT DOWN!")
        sys.exit()

    else:
        # ardışık kareler aynıysa ikincisini kontrol etme
        if center[0] == x_prev and center[1] == y_prev:
            continue
        x_prev = center[0]
        y_prev = center[1]

    # kuyruk çiz (kullanıyoruz), center noktasını pts dizisine ekle
    pts[pts_index].append(center)

    # pts.length kadar dön ve kırmızı nokta çiz (geçmiş karelerdeki noktaları da çizer)
    for j in range(1, len(pts[pts_index])):
        if pts[pts_index][j-1] is None or pts[pts_index][j] is None:
            continue
        cv2.circle(img, pts[pts_index][j], 1, (0, 0, 255), 4)


    if (is_init_frame == True):
        prev_frame_objects.append([(center[0], center[1]), ot.get_init_index(), 0, deque(), -1, 0])
    else:
        cur_frame_objects.append([(center[0], center[1]), 0, 0, deque(), -1, 0])


    if (is_init_frame == False):
        # We only run when we have had at least 1 object detected in the previous (initial) frame
        if (len(prev_frame_objects) != 0):
            cur_frame_objects = ot.sort_cur_objects(prev_frame_objects, cur_frame_objects)


    #                     0             1  2    3       4  5
    # point => [(center[0], center[1]), 0, 0, deque(), -1, 0]
    # 0: center koordinatları
    # 1: id
    # 2: obje kaç frame'dir tespit ediliyor
    # 3: deque
    # 4: (eşlenme verisi) herhangi bir prev obj ile eşlenip eşlenmediği verisi (yada eşleşmiş olduğu obj'nin indexi olabilir)
    # 5: Magnitude of object

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
            p.print("DELTA = " + str(delta) + " || MAG = " + str(vector_mag))

            cv2.putText(img, str("ivme: " + "{:.2f}".format(delta)), (int(tbox[0]), int(tbox[1] - 20)), font, 1, (235, 55, 55), 2, cv2.LINE_AA)
            if delta > 11:
                cv2.putText(img, str("{:.2f}".format(delta)), (int(tbox[0]) + 80 , int(tbox[1])), font, 1, (255, 50, 50), 2, cv2.LINE_AA)

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

    if (is_crash_detected == True):
        cv2.putText(img, f"CRASH DETECTED", (0, 30), font, 1, (0, 0, 255), 2, cv2.LINE_AA) #draw_frame
        p.print(" KAZA OLDUUUUUUUUUUUUUUUUUUUUUUUUUUU ")
        crash_flag = True
        cv2.putText(img, f"CRASH DETECTED", (300, 300), font, 1, (0, 0, 255), 2, cv2.LINE_AA) #draw_frame

    if crash_flag == True:
        cv2.putText(img, f"CRASH DETECTED", (300, 30), font, 1, (0, 0, 255), 2, cv2.LINE_AA) #draw_frame


    if (is_init_frame == False):
        prev_frame_objects = cur_frame_objects.copy()
        cur_frame_objects = []
    is_init_frame = False

    cv2.imshow('YENI FAST ROI', img)

    # yavaş
    # time.sleep(0.8)

    # orta yavaş
    # time.sleep(0.5)

    # orta hızlı
    # time.sleep(0.3)

    # hızlı
    time.sleep(0.15)

    # çok hızlı
    # time.sleep(0.05)

    if cv2.waitKey(1) == ord('q'):
        break
vid.release()
# out.release()
cv2.destroyAllWindows()
