import cv2
from darkflow.net.build import TFNet
import numpy as np
import time

option1 = {
    'model': 'cfg/tiny-yolo-voc-nofight.cfg',
    'load': 16200,
    'threshold': 0.05,
    'gpu': 1.0
}
option2 = {
    'model': 'cfg/tiny-yolo-voc-1c.cfg',
    'load': 35250,   
    'threshold': 0.15,
    'gpu': 1.0
}


tfnet1 = TFNet(option1)
tfnet2 = TFNet(option2)


fc = 30.0
capture = cv2.VideoCapture('testvideo/fighttest25.mp4')
capture.set(3, 720) 
capture.set(4, 1080) 
codec = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X') 
writer = cv2.VideoWriter('testvideo/0612resultt25.avi', codec, fc, (int(capture.get(3)), int(capture.get(4))))

count1 = 0 
count2 = 0 
colors = [tuple(255 * np.random.rand(3)) for i in range(5)]

while (capture.isOpened()):
    stime = time.time()
    ret, frame = capture.read()
    
    if ret:
        results1 = tfnet1.return_predict(frame)
        results2 = tfnet2.return_predict(frame)

        for color, result in zip(colors, results2):

	        tl = (result['topleft']['x'], result['topleft']['y'])
	        br = (result['bottomright']['x'], result['bottomright']['y'])
	        label2 = 'fight'
	        confidence = result['confidence']
	        text = '{}: {:.0f}%'.format(label2, confidence * 100)
	        frame = cv2.rectangle(frame, tl, br, (0,0,250), 7)
	        frame = cv2.putText(frame,  text,tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 250, 0), 2)
	        count2=count2+1


        for color, result in zip(colors, results1):
	        tl = (result['topleft']['x'], result['topleft']['y'])
	        br = (result['bottomright']['x'], result['bottomright']['y'])
	        label1 = 'no_fight'
	        confidence = result['confidence']
	        text = '{}: {:.0f}%'.format(label1, confidence * 100)
	        frame = cv2.rectangle(frame, tl, br, (250,0,0), 7)
	        frame = cv2.putText(frame,  text,tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 250, 0), 2)
	        count1=count1+1    	


	            # confidence = result['confidence']
	            # text = '{}: {:.0f}%'.format(label2, confidence * 100)
	            # frame = cv2.rectangle(frame, tl, br, (0,0,250), 7)
	            # frame = cv2.putText(frame,  text,tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 250, 0), 2)
	            # count2=count2+1	
        
        cv2.imshow('frame', frame)
        writer.write(frame)
        
            
        print('FPS {:.1f}'.format(1 / (time.time() - stime)))
        print(count1)
        print(count2)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        capture.release()
        writer.release()
        cv2.destroyAllWindows()
        break