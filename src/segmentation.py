from word_detector import prepare_img, detect, sort_line
import matplotlib.pyplot as plt
import cv2


def get_segments(image):
    img = prepare_img(cv2.imread(image),50)

    detections = detect(img,kernel_size=25,sigma=11,theta=7,min_area=100)

    line = sort_line(detections)[0] 



    images=[]
    for i, word in enumerate(line):
        images.append(word.img)
    return images
