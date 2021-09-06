from flask import Flask, render_template, Response
from camera import VideoCamera
import cv2


# Flask app
app = Flask(__name__)

camera = None

# This is the decorator for defining the API endpoint
# This function will be called when 'http://localhost:port/' is hit
@app.route('/')
def index():
    return render_template('index.html')

# Video stream generator
def gen():
    global camera
    while True:
    # Get frame from stream (predicted)
        if camera:
            frame = camera.get_frame()
            if frame is None:
            	break

            # For sending as response
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


# API for streaming video from browser to server and emotion detection
@app.route('/video_feed')
def video_feed():
    global camera
    camera = VideoCamera()
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# API to stop the video
@app.route('/stop_video')
def stop_video():
    print('Stopping video...')
    global camera
    camera.stop_video()
    return Response()

# Start the flask server in debug mode at given host
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)