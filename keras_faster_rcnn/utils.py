import cv2
import numpy as np

PROCESS_OUTPUT_DIR = 'process_output/'
dump_process_img = lambda name, img, out=PROCESS_OUTPUT_DIR: cv2.imwrite(out + name + '.jpg', img)

def show_rpn_input_img(input_img):
    """shape: (1, w, h, 1)"""
    cv2.namedWindow('rpn_input', cv2.WINDOW_NORMAL)
    cv2.imshow('rpn_input', np.squeeze(input_img))
    cv2.waitKey(0)

def dump_rpn_input_img(input_img, count = 0):
    """shape: (1, w, h, 1)"""
    img = np.squeeze(input_img)
    dump_process_img(str(count), img)

def draw_rect(img, a, b, show=True):
    cv2.rectangle(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), (0 , 255, 0), 2)
    if show:
        cv2.namedWindow('anchor', cv2.WINDOW_NORMAL)
        cv2.imshow('anchor', img)
        cv2.waitKey(0)

def int_tuple(*args):
    return tuple([int(i) for i in args])

if __name__ == "__main__":
    print(int_tuple(1.2, 1.333))