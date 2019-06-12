import cv2
from darkflow.net.build import TFNet
import numpy as np
import time

option = {
    'model': 'cfg/tiny-yolo-voc-1c.cfg',
    'load': 5400,
    'threshold': 0.25,
    'gpu': 1.0
}

tfnet = TFNet(option)
fc=30.0
capture = cv2.VideoCapture('testvideo/fighttest26.mp4')
capture.set(3, 720) 
capture.set(4, 1080) 
codec = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X') 
writer = cv2.VideoWriter('outputttttt.avi', codec, fc, (int(capture.get(3)), int(capture.get(4))))

count = 0 
colors = [tuple(255 * np.random.rand(3)) for i in range(5)]

while (capture.isOpened()):
    stime = time.time()
    ret, frame = capture.read()
    writer.write(frame)
    if ret:
        results = tfnet.return_predict(frame)
        for color, result in zip(colors, results):
            tl = (result['topleft']['x'], result['topleft']['y'])
            br = (result['bottomright']['x'], result['bottomright']['y'])
            label = result['label']
            confidence = result['confidence']
            text = '{}: {:.0f}%'.format(label, confidence * 100)
            frame = cv2.rectangle(frame, tl, br, color, 7)
            frame = cv2.putText(frame,  text,tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
            count=count+1
        cv2.imshow('frame', frame)
        
        
            
        print('FPS {:.1f}'.format(1 / (time.time() - stime)))
        print(count)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    else:
        capture.release(1)
        writer.release()
        cv2.destroyAllWindows()
        break