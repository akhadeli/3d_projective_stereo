import cv2

cam = cv2.VideoCapture(0, cv2.CAP_FIREWIRE)
cv2.namedWindow("test")
img_counter = 0

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ASCII:ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # ASCII:SPACE pressed
        path = 'images/extraimgs/'
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(path+img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()
cv2.destroyAllWindows()