import cv2


frame_id = 55

img = cv2.imread(f'./dataset/images/{frame_id:06d}.png')

with open(f'./dataset/image-labels/{frame_id:06d}.txt') as fp:
    for line in fp:
        data = line.strip().split(' ')
        x_min, x_max, y_min, y_max = map(int, data[:4])
        cls = data[4]

        cv2.putText(img, cls, (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.line(img, (x_min,y_min), (x_max,y_min), (0,0,255, 255), 1)
        cv2.line(img, (x_min,y_max), (x_max,y_max), (0,0,255, 255), 1)
        cv2.line(img, (x_min,y_min), (x_min,y_max), (0,0,255, 255), 1)
        cv2.line(img, (x_max,y_min), (x_max,y_max), (0,0,255, 255), 1)

cv2.imshow('Frame', img)
cv2.waitKey(0)