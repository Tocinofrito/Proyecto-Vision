# OpenCV Facial Recognition
# Written + Documented by Abdullah Khan

# face_recognize.py
# script used to: train a model on people's faces and recognize the faces using the client-side webcam stream

size = 4
import cv2, sys, numpy, os

# change the paths below to the location where these files are on your machine
# haar_file path
haar_file = 'haarcascade_frontalface_default.xml'
# path to the main faces directory which contains all the sub_datasets
datasets = 'faces'

#############################################################
#Pruebas de gradiente video tiempo real
#Kernel Sobel (Marcaré con "KS")
hx = numpy.array([[-1,0,1],[2,0,2],[-1,0,1]])/8
hy = numpy.array([[-1,-2,-1],[0,0,0],[1,2,1]])/8


print('Training classifier...')
# Create a list of images and a list of corresponding names along with a unique id
(images, labels, names, id) = ([], [], {}, 0)
for (subdirs, dirs, files) in os.walk(datasets):
    for subdir in dirs:
	# the person's name is the name of the sub_dataset created using the create_data.py file
        names[id] = subdir
        subjectpath = os.path.join(datasets, subdir)
        for filename in os.listdir(subjectpath):
            path = subjectpath + '/' + filename
            label = id
            images.append(cv2.imread(path, 0))
            labels.append(int(label))
        id += 1
(width, height) = (130, 100)

# https://www.life2coding.com/drawing-fancy-round-rectangle-using-opencv-python/
#Se crea un tipo cuadro alrededor del rostro
def rounded_rectangle(img, pt1, pt2, color, thickness, r, d):
    x1,y1 = pt1
    x2,y2 = pt2
 
    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
 
    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
 
    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
 
    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

# Create a numpy array from the lists above
(images, labels) = [numpy.array(lists) for lists in [images, labels]]

# OpenCV trains a model from the images using the Local Binary Patterns algorithm
model = cv2.face.LBPHFaceRecognizer.create()
# train the LBP algorithm on the images and labels we provided above
model.train(images, labels)

face_cascade = cv2.CascadeClassifier(haar_file)
webcam = cv2.VideoCapture(0)
print('Classifier trained!')
print('Attempting to recognize faces...')
while True:
    (_, im) = webcam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # detect faces using the haar_cacade file
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        # colour = bgr format
	# draw a rectangle around the face and resizing/ grayscaling it
	# uses the same method as in the create_data.py file
        # cv2.rectangle(im,(x,y),(x + w,y + h),(0, 255, 255),2)
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width, height))
        # try to recognize the face(s) using the resized faces we made above
        prediction = model.predict(face_resize)
        rounded_rectangle(im, (x, y), (x + w, y + h), (0, 255, 255), 2, 15, 30)
        # if face is recognized, display the corresponding name
        if prediction[1] < 74:
            cv2.putText(im,'%s' % (names[prediction[0]].strip()),(x + 5, (y + 25) + h), cv2.FONT_HERSHEY_PLAIN,1.5,(20,185,20), 2)
            # print the confidence level with the person's name to standard output
            confidence = (prediction[1]) if prediction[1] <= 100.0 else 100.0
            print("predicted person: {}, confidence: {}%".format(names[prediction[0]].strip(), round((confidence / 74.5) * 100, 2)))
        # if face is unknown (if classifier is not trained on this face), show 'Unknown' text...
        else:
            cv2.putText(im,'Unknown',(x + 5, (y + 25) + h), cv2.FONT_HERSHEY_PLAIN,1.5,(65,65, 255), 2)
            print("predicted person: Unknown")

    # show window and set the window title
    #KS
    # Aplicando filtro gauss,x, y, xy
    im = cv2.GaussianBlur(im, (5,5),0)
    a = cv2.filter2D(im, -1, hx)
    b = cv2.filter2D(im, -1, hy)
    c = cv2.filter2D(a, -1, hy)

# Aplicando magnitud a las matrices
    a = a ** 2
    b = b ** 2
    G = numpy.sqrt(a + b)

# Escalamos los valores máximos y mínimos de G a 255
    G = (G / G.max() * 255).astype(numpy.uint8)
    
    cv2.imshow('OpenCV Face Recognition -  esc to close', G)
    key = cv2.waitKey(10)
    # esc to quit applet
    if key == 27:
        break
