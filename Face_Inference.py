import cv2 as cv
import paho.mqtt.client as paho

#mqtt broker setup
broker = "127.0.0.1"
port = 18831
client1= paho.Client("control1")
client1.connect(broker,port)

#camera setup
cap = cv.VideoCapture(0)
width = 128
height = 128
cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)

# Load the model.
net = cv.dnn.readNet('model.xml',
                    'model.bin')

# Specify target device.
net.setPreferableTarget(cv.dnn.DNN_TARGET_MYRIAD)

# class name
CLASSES = ["06_none","06_have","18_none","18_have","30_none","30_have"
           ,"other_none","other_have"]

# Read an image.
while(True):
    ret, frame = cap.read()
    
    if frame is None:
        raise Exception('Image not found!')
        
    # Prepare input blob and perform an inference.
    blob = cv.dnn.blobFromImage(frame, size=(256, 256), ddepth=cv.CV_8U)
        
    net.setInput(blob)
    out = net.forward()
    
    #print('Weights :', out)
        
    label = 0
    max_num = -1
        
    for i in range(8):
        if out.item(i) > max_num:
            max_num = out.item(i)
            label = i
        
    text = 'Label   : ' + CLASSES[label] + ' , ' + str(label)
    print(text)
    print()
    
    #send message to mqtt broker
    ret = client1.publish("sensor/camera",label)
    
    cv.imshow('frame' , frame)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv.destroyAllWindows()
