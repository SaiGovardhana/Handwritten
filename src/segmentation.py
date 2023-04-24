from word_detector import prepare_img, detect, sort_line
import matplotlib.pyplot as plt
import cv2

# (1b) scale to specified height because algorithm is not scale-invariant
def get_segments(image):
    img = prepare_img(cv2.imread(image),50)

    # (2) detect words in image
    detections = detect(img,
                        kernel_size=25,
                        sigma=11,
                        theta=7,
                        min_area=100)

    # (3) sort words in line
    line = sort_line(detections)[0] 

    # (4) show word images

    images=[]
    for i, word in enumerate(line):
        images.append(word.img)
    return images
