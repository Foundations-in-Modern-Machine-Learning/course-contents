import cv2
import numpy as np
import dlib
from torchvision import transforms
from PIL import Image
import torch
import torchvision
import torch.nn as nn


# Create model object
model = torchvision.models.vgg19(pretrained=True)
    
# Change the output layer to output 7 classes instead of 1000 classes
num_ftrs = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_ftrs, 7)

# Load trained model weights
model.load_state_dict(torch.load('trained_model.pt', map_location='cpu'))


font = cv2.FONT_HERSHEY_SIMPLEX
img_h, img_w = 48, 48


# Helper function to convert rectangle coordinates to bounding box coordinates
def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)

# Function to detect face and plot bounding box around it
def detect_face(image, return_bb=False):
    #### YOUR CODE HERE ####

    if return_bb:
        return image[y:y+h, x:x+w], (x, y, w, h)

    return image[y:y+h, x:x+w]


# Helper function to run inference on model 
def predict_emotion(model, x):
    #### YOUR CODE HERE  ####

    return classes[preds.cpu().numpy()[0]]


# Video camera class to handle camera functions
class VideoCamera(object):
    def __init__(self):
        # Pass path to the video file you want to use
        self.video = cv2.VideoCapture('path_to_video_file') 

        # If you want to use webcam, uncomment this line and comment out previous one
        # self.video = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # Destructor
    def __del__(self):
        self.video.release()
        cv2.destroyAllWindows()

    # returns camera frames along with bounding boxes and predictions
    def get_frame(self):
        success, fr = self.video.read()
        if not success:
           return None
        
        face, coords = detect_face(fr, return_bb=True)
        print('Read new frame')

        # Run inference on detected face else return the image frame as it is
        if face is not None:
            face = Image.fromarray(face)
            transform = transforms.Compose([transforms.Resize((img_h, img_w)), transforms.ToTensor()])
            ip = transform(face).view((-1, 3, img_h, img_w))
            pred = predict_emotion(model, ip.float())

            x, y, w, h = coords
            cv2.putText(fr, pred, (x, y), font, 1, (255, 255, 0), 2)
            cv2.rectangle(fr,(x,y),(x+w,y+h),(255,0,0),2)

        _, jpeg = cv2.imencode('.jpg', fr)
        return jpeg.tobytes()


    # Stop video stream
    def stop_video(self):
        self.video.release()
        cv2.destroyAllWindows()