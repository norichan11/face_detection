import dlib
import cv2

# 選擇第一支攝影機
cap = cv2.VideoCapture(0)
while(True):
  # 從攝影機擷取一張影像
  ret, frame = cap.read()
  # 顯示圖片
  cv2.imshow('frame', frame)
  cv2.imwrite("cameraoutput.jpg", frame)

  # 讀取照片圖檔
  img = cv2.imread('cameraoutput.jpg')
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  # Dlib 的人臉偵測器
  detector = dlib.get_frontal_face_detector()

  # 偵測人臉
  face_rects = detector(img, 0)

  # 取出所有偵測的結果
  for i, d in enumerate(face_rects):
    x1 = d.left()
    y1 = d.top()
    x2 = d.right()
    y2 = d.bottom()

    # 以方框標示偵測的人臉
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 4, cv2.LINE_AA)
    cv2.imwrite("dataset\\guest.jpg", gray[y1:y2, x1:x2])

  # 若按下 q 鍵則離開迴圈
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

  # 顯示結果
  cv2.imshow("Face Detection", img)

# 釋放攝影機
cap.release()
# 關閉所有 OpenCV 視窗
cv2.destroyAllWindows()
