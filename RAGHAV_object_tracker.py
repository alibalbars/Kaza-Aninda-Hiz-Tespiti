from MY_print_methods import print_class
p = print_class()

"""
OBJECT TRACKING IMPLEMENTATION

It is extremely important to properly track objects across subsequent frames because by default, objects
do not necessarily retain their original indexes across frames. I.E. It is entirely possible that an object
with index 0 in the intitial frame is detected with index 1 in the next frame.

Our solution to this is fairly simple, but is nonetheless effective. In essense the pseudo code looks something
like this:

1 For every new object detected in the current frame:
    1.1 For every old object from the previous frame:
        1.1.1 Find an old object such that the distance between this old object and new object is the least
              among all combinations of old object and new object pairs.
    1.2 The old object with the least distance to the new object is most certainly the same object. This
        is because an object can only move so far between subsequent frames so this distance will almost
        always be smaller than the distance between 2 different objects.
    1.3 Assign the new object the index of the corresponding old object.
    1.4 Update new object's number of frames detected and vectors accordingly.

However, the above algorithm only works if we assume that the number of objects across two frames will remain
constant. As soon as objects start disappearing and appearing, this algorithm will break down as objects detected
for the first time can be assigned the wrong index.

To solve this, we check if an old object has already been assigned to a new object. If this is the case and there
is a conflict between assigning two new objects to a single old object, we will compare the distances between
the two pairs and decide which new object the old object actually corresponds to. The incorrect new object is
then marked as "not found" and will later be assigned a completely new index.

To facilitate assigning of new indexes, we use a simple set to store the indexes currently in use. To find an unused
index that is currently available, we simply start at index 0 and keep increasing the index until we find the
smallest unused index. This is what the find_next_free_index() function does.

Doing all of the above requires a myriad of information that is stored in prev_frame_objects and cur_frame_objects.
The structure of these lists has been detailed below:

prev_frame_objects structure:
Index 0: Tuple for the midpoint of the object
Index 1: Index of the object, used for unique indentification of objects
Index 2: Number of frames object has been consecutively detected for
Index 3: Deque to store midpoints of the object over the past 5 frames. Used for vector calculation
Index 4: Points to the in-array index of the new object that this old object has been assigned to. This
         is used to fix any conflicts between 2 or more new objects that have been assigned to the same
         old object.
Index 5: Magnitude of object from previous frame

Structure for cur_frame_objects is the same, except that Index 4 is instead used as a flag to determine
whether the object has been assigned to a corresponding old object. This is useful for assigning completely
newly detected objects with an unused index.

OBJECT VECTOR CALCULATION

The vector for every object is calculated from the object's midpoint at the 1st frame, and the midpoint at
the 5th frame. We store these midpoints in a deque inside prev_frame_objects and/or cur_frame_objects. The deque
is kept updated every frame by removing the oldest midpoint and adding the latest midpoint.

Once we've worked out the vector from the deque at an initial frame, we can get the next vector by doing the same
calculation for the updated deque in the next frame. By finding the difference between the magnitude of both
vectors, we can determine whether a crash has occured based on our set threshold.
"""

class object_tracker:
    def __init__(self):
        self.cur_indexes = set() # Keeping track of currently in-use object indexes.
        self.init_index = -1 # Used when assigning indexes to objects in the very first frame

    # This is used for assigning indexes to the objects inside prev_frame_objects during the first ever frame.
    def get_init_index(self):
        self.init_index += 1
        return self.init_index

    # This function returns the smallest unused index that isn't present inside self.cur_indexes.
    def find_next_free_index(self):
        start = 0
        while True:
            if (start in self.cur_indexes) == True:
                start += 1
            else:
                return start

    # Simply gets distance between two points. This is the primary way we determine which points in the
    # prev frame correspond to certain points in the cur frame
    def get_dist(self, prev_point, cur_point):
        y_square = (cur_point[1] - prev_point[1]) ** 2
        x_square = (cur_point[0] - prev_point[0]) ** 2
        return ((x_square + y_square) ** 0.5)

    # This function does the brunt of the object tracking. It returns an updated list of cur_objects which
    # has an updated deque and correctly assigned unique identifier indexes.
    def sort_cur_objects(self, prev_objects, cur_objects):
        # Resetting our set of currently used indexes everytime this function is called.
        self.cur_indexes = set()
        # İçi boş bir set oluşturur
        # NOT: set cannot contains duplicate items, items are not in any particular order.
        # set ancak immutable objeleri içerir => string, number, tuple (mutable: list, dictionary)
        # p.print(cur_indexes)

        for i in range(0, len(cur_objects)): # Iterating through all current frame objects
            min_dist = 99999 # Will store least distance of a particular prev_object to our loop's current new object.
            min_dist_index = None # Will store index of prev_object with least distance

            for j in range(0, len(prev_objects)): # Iterating through all previous frame objects
                # Checking if our new distance calculation is lower than the previous. If it is, we will
                # change our min_dist and min_dist_index accordingly.
                check_dist = self.get_dist(prev_objects[j][0], cur_objects[i][0])
                if (check_dist <= min_dist):
                    min_dist = check_dist
                    min_dist_index = j

            # yukarıda; her bir cur obj için => mevcut cur obj ile tüm prev objelerin uzaklıklarını al
            # en kısa uzaklığa sahip olan prev objenin indexini değişkene kaydet.

            # aşağıda; mevcut cur objemize en yakın olan prev objeyi yukarıda bulmuştuk,
            # bu prev objenin default'ta (ilk eklemede) -1 olarak gelen sondan ikinci olan parametresi
            # -1 değilse işlem gerçekleştir [galiba eşleme]
            # o parametre -1 ise => prev obj henüz eşlenmemiş (?)
            # o parametre -1 den farklıysa => zaten eşlenmiş demek (doğru)

            # This conditional executes in the event that a previous object has been already assigned to a current
            # cur objemizin prev objesi eşlenmiş ise
            if (prev_objects[min_dist_index][4] != -1):
                # Get distance of previous object and the new object that has been assigned to it already
                # (cur obje ile prev obje arası mesafeyi ölç)
                                                                # prev_objects[min_dist_index][4] => -1 muadili rakam

                # birinci par => mevcut cur'un bağlı olduğu prev objesinin center'ı
                # ikinci parametre => (mevcut cur'un bağlı olduğu (?)) prev'in bağlı olduğu cur obj nin center'ı

                # prev_objects[min_dist_index][4] => önceki arabanın bağlı olduğu current araba'nın indexi
                # ilk cur arabanın bir frame de yer değiştirdiği mesafe
                # first_dist => mevcut ve prev aracın arasındaki uzaklık
                first_dist = self.get_dist(prev_objects[min_dist_index][0], cur_objects[prev_objects[min_dist_index][4]][0])
                # Get distance of previous object and the new object that is causing this new conflict (bağdaşmamak)
                second_dist = self.get_dist(prev_objects[min_dist_index][0], cur_objects[i][0])

                # If the new conflicting object actually corresponds to previous object
                if (second_dist <= first_dist):
                    # Add index to set of used indexes
                    self.cur_indexes.add(prev_objects[min_dist_index][1]) # (1) prev index'ini cur_indexes set'ine ekle
                    # p.print("CUR INDEXES (ilk) => " + str(self.cur_indexes))
                    # Assign unique identifier index correctly
                    cur_objects[i][1] = prev_objects[min_dist_index][1] # (2) cur index = prev index
                    # Set previously uncontested new object as not used. This is so that in the event it is an
                    # entirely new detection, we can assign it a new index.
                    cur_objects[prev_objects[min_dist_index][4]][4] = -1 # (3) current'a en yakın prev'in current'ını bağlanmamış olrk işaretle
                    # Set new conflicting object as used.
                    cur_objects[i][4] = min_dist_index # (4) mevcut current'ın eşlenme değerine bağlı olduğu prev'in indexini ata
                    # Make the previous object point to the new conflicting object.
                    prev_objects[min_dist_index][4] = i # (5) current'a bağlı prev'in eşlenme değerine current'ın index'ini ata

                # If previously assigned new object is smaller than conflicting new object.
                # conflict varsa
                else:
                    # Conflicting new object is set as unused.
                    cur_objects[i][4] = -1 # (1) mevcut current'ın eşlenme değerini eşlenmemiş olarak ayarla
                    # p.print("OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")
                    continue # Continuing, because we do not want to mess with this completely new object's deque.

            # This executes when a previous object is unassigned to a new object.
            # (current'a en yakın prev obje eşlenmemiş ise)
            else:
                self.cur_indexes.add(prev_objects[min_dist_index][1]) # (1) prev index'ini cur_indexes set'ine ekle
                # p.print(prev_objects[min_dist_index][1])
                # p.print("CUR INDEXES (son) => " + str(self.cur_indexes))
                cur_objects[i][1] = prev_objects[min_dist_index][1] # Updating index of new object # (2) current ID = prev ID
                prev_objects[min_dist_index][4] = i # Previous object is made to point to new object # (3) prev eşleme değeri = cur index
                cur_objects[i][4] = min_dist_index # Setting new object as used/found. # (4) cur eşleşme değeri = prex index

                # yukarıdaki genel for'da cur ve prevlerin eşlenik değerlerini atadı

        # This loop handles updating the deque along with the previous magnitude of the object (index 5)
        for i in range(0, len(cur_objects)):
            if (cur_objects[i][4] != -1): # If the new object's corresponding previous object has been found
                corres_prev_obj = cur_objects[i][4] # ( corres_prev_obj = current'ın previnin indexi )

                # If the object has been detected for less than 5 frames
                if (prev_objects[corres_prev_obj][2] < 5): #  (prev 5 kareden azdır tespit edilmişse) (prev objenin frame parametresi 5'ten küçükse)
                    cur_objects[i][3] = prev_objects[corres_prev_obj][3].copy() # (1) cur'un dequesi = prev'in dequesi
                    cur_objects[i][3].append(cur_objects[i][0]) # (2) cur'un deque'sine cur'un center'ını ekle
                else: # If object has been detected for at least 5 frames. # (prev en az 5 karedir tespit ediliyorsa)
                    cur_objects[i][3] = prev_objects[corres_prev_obj][3].copy() # (1) cur'un dequesi = prev'in dequesi
                    cur_objects[i][3].popleft() # deque'nin solundaki elemanı çıkar
                    cur_objects[i][3].append(cur_objects[i][0]) # deque'nin sağına center'ı ekle

                    # Örnek deque => deque([(421, 293), (422, 293), (425, 296), (426, 296), (426, 297)])
                    # prev_objects[corres_prev_obj][3][-1][0] => dequenin son elemanı'nın x değeri
                    # prev_objects[corres_prev_obj][3][0][0] => dequnin ilk elamanının x değeri

                    # prev_objects[corres_prev_obj][3][-1][1] => dequenin son elemanının y değeri
                    # prev_objects[corres_prev_obj][3][0][1]] => dequenin ilk elemanının y değeri

                    # type(vector) => list

                    # We make the current object store the previous object's magnitude of vector
                    vector = [prev_objects[corres_prev_obj][3][-1][0] - prev_objects[corres_prev_obj][3][0][0], \
                        prev_objects[corres_prev_obj][3][-1][1] - prev_objects[corres_prev_obj][3][0][1]] # (x, y)
                    vector_mag = (vector[0]**2 + vector[1]**2)**(1/2) # aracın previous kare ile 5 kare öncesi arası uzaklığı (piksel bazında)
                    cur_objects[i][5] = vector_mag # current'a magnitude değerini ata

                # Updating consecutive detection frame count of object
                cur_objects[i][2] = prev_objects[corres_prev_obj][2] + 1 # current'ın tespit edildiği frame sayısını bir arttır

        # This iterates through the cur_frame_objects again and sees which ones have not
        # been assigned an index, and assigns them an unused index accordingly.
        for new_object in cur_objects:
            if new_object[4] == -1: # yeni current eşlenmiş ise
                new_object[1] = self.find_next_free_index() # ID ata
                self.cur_indexes.add(new_object[1]) # ID'yi cur_indexes set'ine ata
                p.print("OOOOOOOOOOOOOOOOOO")
            else: # eşlenmemiş ise
                new_object[4] = -1 # eşlenme verisini -1'e eşitle

        return cur_objects
