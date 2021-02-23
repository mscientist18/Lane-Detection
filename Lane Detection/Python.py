import cv2
import numpy as np


def make_coordinates(image, line_parameters):
    slope, intersept = line_parameters
    y1 = image.shape[0]
    y2 = int((y1 * (3 / 5)))
    x1 = int((y1 - intersept) / slope)
    x2 = int((y2 - intersept) / slope)
    return np.array([x1, y1, x2, y2])


def avg_slope(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intersept = parameters[1]
# lines on left have -ve slope  and right have +ve slope bcause top left is 0,0
        if slope > 0:
            left_fit.append((slope, intersept))
        else:
            right_fit.append((slope, intersept))
    left_fit_avg = np.average(left_fit, axis=0)
    right_fit_avg = np.average(right_fit, axis=0)
    left_line = make_coordinates(image, left_fit_avg)
    right_line = make_coordinates(image, right_fit_avg)
    return np.array([right_line, left_line])


def canny(img):
    grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(grey, (5, 5), 0)
    cn = cv2.Canny(blur, 50, 150)
    return cn


def display_lines(img, lines):
    lineimage = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line
            cv2.line(lineimage, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return lineimage


def roi(img):  # region of interest
    h = img.shape[0]
    pl = np.array([  # array of polygons ie 1 triangle
        [(200, h), (1100, h), (550, 250)]
    ])
    mask = np.zeros_like(img)  # create array of 0s of Img
    cv2.fillPoly(mask, pl, 255)
    msk = cv2.bitwise_and(img, mask)
    return msk


# image = cv2.imread('tsi.jpg')
# frame = np.copy(image)
# cn = canny(frame)
# cropped_image = roi(cn)
# lines = cv2.HoughLinesP(
#     cropped_image,
#     2,
#     np.pi / 180,
#     100,
#     np.array([]),
#     minLineLength=40,
#     maxLineGap=5)
# avg = avg_slope(frame, lines)
# line_image = display_lines(frame, avg)
# real_Image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
# cv2.imshow("mk", real_Image)
# cv2.waitKey(0)

cap = cv2.VideoCapture("test.mp4")
while (cap.isOpened()):
    _, frame = cap.read()
    cn = canny(frame)
    cropped_image = roi(cn)
    lines = cv2.HoughLinesP(
        cropped_image,
        2,
        np.pi / 180,
        100,
        np.array([]),
        minLineLength=40,
        maxLineGap=5)
    avg = avg_slope(frame, lines)
    line_image = display_lines(frame, avg)
    real_Image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    cv2.imshow("mk", real_Image)
    if cv2.waitKey(1) is ord('q'):  # press q to quit
        cap.release()
        cv2.destroyAllWindows()
