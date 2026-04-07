import cv2

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()

    if not success:
        print("Camera not working")
        break

    cv2.imshow("Cam Test", img)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()