import cv2
MIN_MATCHES = 15
model = cv2.imread('bridgepic.jpg',0)
cap = cv2.VideoCapture(0)

# Initiate ORB detector
orb = cv2.ORB_create()

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# compute the descriptors with ORB
kp_model, des_model = orb.detectAndCompute(model, None)  

while True:
    _, frame = cap.read()
    grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp_frame, des_frame = orb.detectAndCompute(grayframe, None)
    matches = bf.match(des_model, des_frame)
    matches = sorted(matches, key=lambda x: x.distance)
    if len(matches) > MIN_MATCHES:
        frame = cv2.drawMatches(model, kp_model, frame, kp_frame, matches[:MIN_MATCHES], 0, flags=2)
        cv2.imshow('frame', frame)
    key = cv2.waitKey(1)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

        

 
# draw only keypoints location,not size and orientation
#img2 = cv2.drawKeypoints(img, kp, img, color=(0,255,0), flags=0)
#cv2.imshow('keypoints',img2)
#cv2.waitKey(0)