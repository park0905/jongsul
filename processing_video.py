import cv2
from darkflow.net.build import TFNet
import numpy as np
import time

option = {
    'model': 'cfg/tiny-yolo-voc-1c.cfg',
    'load': 36000, ##13625
    'threshold': 0.20,
    'gpu': 1.0
}


# option2 = {
#     'model': 'cfg/tiny-yolo-voc-2c.cfg',
#     'load': 13625, ##13625
#     'threshold': 0.20,
#     'gpu': 1.0
# }

# option = {
#     'model': 'cfg/tiny-yolo-voc-3c.cfg',
#     'load': 13625, ##13625
#     'threshold': 0.20,
#     'gpu': 1.0
# }










tfnet = TFNet(option)
## tfnet2 = TFNet(option2)

capture = cv2.VideoCapture('testvideo/fighttest34.mp4')
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
writer = cv2.VideoWriter('output111111t.avi', fourcc, 35.0, (640, 480))

count = 0 
colors = [tuple(255 * np.random.rand(3)) for i in range(5)]

while (capture.isOpened()):
    stime = time.time()
    ret, frame = capture.read()
   
    if ret:
        results = tfnet.return_predict(frame)
      ## resultS2 =tfnet2return_predict(frame)
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
        writer.write(frame)
        
            
        print('FPS {:.1f}'.format(1 / (time.time() - stime)))
        print(count)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
    else:
        capture.release()
        writer.release()
        cv2.destroyAllWindows()
        break