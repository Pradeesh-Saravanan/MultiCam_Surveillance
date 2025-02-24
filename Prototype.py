#Motion detection
import cv2
import time
import numpy as np 
class MotionDetector:
    def __init__(self,src = 0,min_contour_area = 1000,reset_interval=10):
        self.src = src
        self.cap = cv2.VideoCapture(0)
        self.first_frame = None

        #Motion detection
        self.min_contour_area = min_contour_area
        self.reset_interval = reset_interval
        self.last_reset_time = time.time()
        
        #Object tracking 
        self.trackers = cv2.legacy.MultiTracker_create()
        self.objects = []

    def initialize_first_frame(self):
        ret,frame = self.cap.read()
        if not ret :
            print("couldn't read frame")
            return False
        self.first_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        self.first_frame = cv2.GaussianBlur(self.first_frame,(21,21),0)
        return True
    def use_Method(self,contours,method,frame):
        for contour in contours:
                if cv2.contourArea(contour)<self.min_contour_area:
                    continue
                if(method=="rectangle"):
                    #Rectangle bounding
                    (x,y,w,h) = cv2.boundingRect(contour)
                    # cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                    #Neon blue color
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(255, 217, 102),2)
                elif(method=="hull"):
                    # For irreguar shapes like human,animals
                    hull = cv2.convexHull(contour)
                    cv2.drawContours(frame, [hull], 0, (255, 0, 0), 2)
                elif(method=="rotated"):
                    #For moving objects like cars or tilted objects 
                    rect = cv2.minAreaRect(contour)
                    box = cv2.boxPoints(rect)
                    box = np.intp(box)
                    cv2.drawContours(frame, [box], 0, (0, 255, 255), 2)
                elif(method=="outline"):
                    cv2.drawContours(frame, [contour], -1, (0, 0, 255), 2)        
    def detect_motion(self):
        if not self.initialize_first_frame():
            return
        while True:
            ret,frame = self.cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray,(21,21),0)
            delta_frame = cv2.absdiff(self.first_frame,gray)
            thresh = cv2.threshold(delta_frame,30,255,cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh,None,iterations=2)


            if time.time() - self.last_reset_time > self.reset_interval:
                self.first_frame = gray
                self.last_reset_time = time.time()
                continue
            contours,_ = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            
            #options: rectangle, hull, rotated, outline
            self.use_Method(contours,"rectangle",frame)

            height, width, _ = frame.shape
    
            # Text to display
            system_status = "System Status: Active"
    
            # Get the size of the text to properly align it
            text_size = cv2.getTextSize(system_status, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    
            # Calculate the position for the top-right corner
            text_x = width - text_size[0] - 10  # 10 pixels from the right edge
            text_y = 40  # 40 pixels from the top

            # Overlay the text in the top-right corner
            cv2.putText(frame, system_status, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)


            # cv2.putText(frame, "System Status: Active", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("Cam1",frame)
            self.first_frame = gray
            if cv2.waitKey(1) & 0xFF ==ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()
    
    def update_trackers(self,frame):
        success,boxes = self.trackers.update(frame)
        return boxes if success else []
    def add_new_objects(self,frame,objects):
        for obj in objects:
            x,y,w,h = obj
            tracker = cv2.legacy.TrackerKCF_create()
            self.trackers.add(tracker,frame,(x,y,w,h))
    def track_objects(self):
        if not self.initialize_first_frame():
            return
        while True:
            ret,frame = self.cap.read()
            if not ret:
                break
            new_objects = []
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray,(21,21),0)

            delta_frame = cv2.absdiff(self.first_frame,gray)
            thresh = cv2.threshold(delta_frame,30,255,cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh,None,iterations=2)

            contours,_ = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) < self.min_contour_area:
                    continue
                x,y ,w,h = cv2.boundingRect(contour)
                new_objects.append((x,y,w,h))
            

            tracked_objects = self.update_trackers(frame)
            self.add_new_objects(frame,new_objects)

            overlay = frame.copy()
            for box in tracked_objects:
                x,y,w,h = [int(v) for v in box]
                cv2.rectangle(overlay,(x,y),(x+w,y+h),(0,255,0),thickness=cv2.FILLED)

            alpha = 0.4
            frame = cv2.addWeighted(overlay,alpha,frame,1-alpha,0)

            for i,box in enumerate(tracked_objects):
                x,y,w,h = [int(v) for v in box]
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),1)
                cv2.putText(frame,f"ID {i+1}",(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),1)
            
            cv2.imshow("Object Tracking",frame)
            self.first_frame = gray
            if cv2.waitKey(1) & 0xFF==ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = MotionDetector(src = 0,min_contour_area=1000,reset_interval=100)
    detector.detect_motion()
    # detector.track_objects()