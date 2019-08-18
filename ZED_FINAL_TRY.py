import pyzed.sl as sl #The library responsible for accessing the facilities of the ZED camera is being imported
import cv2 #The opencv library is being imported
import numpy as np #The numpy library is being imported
import struct #The struct library is being imported
import math #The math library is being imported
import serial

height = 0.82 #The height of the ZED camera is hard-coded
length = 0.73 #The distance between the laser module and the ZED Camera is hard-coded
ser = serial.Serial('COM7', 9600, writeTimeout=0)

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
"sofa", "train", "tvmonitor"]#In the video feed, an attempt is made to recognize the above listed objects. The objects are called as classes.

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3)) # Set colors variable to random value

net = cv2.dnn.readNetFromCaffe(r'C:\Users\brsri\Desktop\MobileNetSSD_deploy.prototxt.txt',r'C:\Users\brsri\Desktop\MobileNetSSD_deploy.caffemodel') #Initializing prototxt and frozen weights of pre-trained Caffe model

def Click(event,x,y,flags,param): #Once the user touches the screen, this function is called
    mouseX = x #The pixel coordinates of the pixel which has been touched by the user has been assigned to mouseX and mouseY
    mouseY = y
    for i in l: #Iterating over the list containing the extermities of the identified objects
        if(event == cv2.EVENT_LBUTTONDOWN and mouseX >= i[0] and mouseX <= i[2] and mouseY >= i[1] and mouseY <= i[3]): #It is checked if the point touched by the user lies within the identified object
            print("Clicked an Object")
            print("Centre of the object is: " + str((i[0]+i[2])/2)+','+str((i[1]+i[3])/2))
            centx = (int(i[0]+i[2])/2) #The x coordinate of the center of the object is found
            centy = int((i[1]+i[3])/2) #The y coordinate of the center of the object is found
            err, point_cloud_value = point_cloud.get_value(centx,centy) #The real world coordinates corresponding to the centroid of the object is obtained as a list
            X = (float(round(point_cloud_value[0])))/1000 #The x coordinate is obtained from the list 
            Y = (float(round(point_cloud_value[1])))/1000 #The y coordinate is obtained from the list
            Z = (float(round(point_cloud_value[2])))/1000 #The z coordinate is obtained from the list
            print(X,Y,Z,type(X),type(Y),type(Z))
            phi = (180/math.pi)*(math.asin(Z/(math.sqrt((float(length)-(X))**2+Z**2)))) #The yaw angle of the first servo motor is calculated
            theta = (180/math.pi)*(math.atan(abs(height-Y)/math.sqrt((float(length)-(X))**2+Z**2)))#The pitch angle of the second servo motor is calculated
            print(phi,theta)
        mouseX,mouseY = x,y

zed = sl.Camera() #The camera feature has been alloted to the 'zed' variable.  
init = sl.InitParameters() #The intitial parameters of the ZED has been assigned to 'init' variable
init.camera_resolution = sl.RESOLUTION.RESOLUTION_HD2K #The resolution of the camera has been fixed to HD2K
init.depth_mode = sl.DEPTH_MODE.DEPTH_MODE_PERFORMANCE #The depth mode of the camera is set to 'Performance'  
init.coordinate_units = sl.UNIT.UNIT_MILIMETER #The units of coordinates of the points in the point cloud is set to millimeteres   

err = zed.open(init)#An attempt is made to access the camera 
if err != sl.ERROR_CODE.SUCCESS : #It is checked whether accessing the camera was unsuccessfull
        print(repr(err)) #If the camera couldn't be accessed, then the       
        zed.close() #The camera feature is cut off
        exit(1) #The code is exited
        
runtime = sl.RuntimeParameters() #The runtime parameters of the ZED has been assigned to 'runtime' variable
runtime.sensing_mode = sl.SENSING_MODE.SENSING_MODE_STANDARD #The sensing mode of the camera is set to 'Standard'

# Prepare new image size to retrieve full-resolution images
image_size = zed.get_resolution() #The resolution is assigned to a variable
new_width = image_size.width #The width of the image is obtained
new_height = image_size.height #The height of the image is obtained

# Declare your sl.Mat matrices
image_zed = sl.Mat(new_width, new_height, sl.MAT_TYPE.MAT_TYPE_8U_C4)
point_cloud = sl.Mat()
#The above 2 matrices are declared as per the image size and with the appropriate data type
key = ' '
while key != 113 :
    err = zed.grab(runtime) #An attempt is made to access the runtime parameters 
    if err == sl.ERROR_CODE.SUCCESS : #If the above doesn't result in an error...
        # Retrieve the left image in full resolution
        zed.retrieve_image(image_zed, sl.VIEW.VIEW_LEFT, sl.MEM.MEM_CPU, int(new_width), int(new_height))
        # Retrieve the RGBA point cloud in full resolution
        zed.retrieve_measure(point_cloud, sl.MEASURE.MEASURE_XYZRGBA, sl.MEM.MEM_CPU, int(new_width), int(new_height))
        
        # To recover data from sl.Mat to use it with opencv, use the get_data() method
        # It returns a numpy array that can be used as a matrix with opencv
        image_ocv = image_zed.get_data()
        image_ocv = cv2.cvtColor(image_ocv,cv2.COLOR_BGRA2BGR)
        (h, w) = image_ocv.shape[:2]

        #Variable to hold images of same spatial dimension and pre-processed images
        blob = cv2.dnn.blobFromImage(cv2.resize(image_ocv, (300, 300)), 0.007843, (300, 300), 127.5)
        #Pass the image collection into the network
        net.setInput(blob)
        #Extraction of confidence values
        detections = net.forward()

        global l
        l = []

        for i in np.arange(0, detections.shape[2]): #Iterating through predictions
                idx = int(detections[0, 0, i, 1])
                #Create bounding boxes
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h]) 
                (startX, startY, endX, endY) = box.astype("int") 
                cv2.rectangle(image_ocv, (startX, startY), (endX, endY),COLORS[idx], 2)
                param2 = l.append([startX,startY,endX,endY]) #Corner points updation
        
        cv2.imshow("Image", image_ocv)
        #Setting mouseclick event
        cv2.setMouseCallback("Image",Click,param = l)
        key = cv2.waitKey(10)


cv2.destroyAllWindows()
zed.close()
