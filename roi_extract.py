
import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
import random
# file_name = 'XY_20090222_F1_A_F_F_E2_1F_E2_E2_E2_1E_E1_E2_E_yiyanzhe.jpg'
file_name = 'XX_19920919_I_H_I_I_I_I_I_I_I_I_I_I_I_pancan.jpg'
img = cv2.imread('data/' + file_name)

# 配置常量
PROCESS_OUTPUT_DIR = 'process_output/'
LOG_PROCESS_IMG = True
## 阈值
BINARY_THRESH = 40 # 二值化
K_THRESH = (-2, 2)  # 选取直线斜率
PARALLEL_THRESH = 10 # 平行线斜率判定
LINE_LEN_THRESH = 50 # 直线长度
CLOSER_PARALLEL_LINE_THRESH = 50 # 线段端点距离极差

# Helper
dump_process_img = lambda name, img: cv2.imwrite(PROCESS_OUTPUT_DIR + name + '.jpg', img) if LOG_PROCESS_IMG else ''
def print_lines(img, name, lines, color = 255):
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), color, 2)
    dump_process_img(name, img)
def cal_line_len(x1, y1, x2, y2):
    return math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))

# 基础方法
# 转灰度图
def cvt_gray(img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# 二值化
def cvt_binary(img):
    ret, binary_img = cv2.threshold(gray_image, BINARY_THRESH, 255, cv2.THRESH_BINARY)
    return binary_img
# 降噪 中值滤波
def blur(img):
    return cv2.medianBlur(img, 5)
# 边缘检测
def edge_detection(binary_img):
    return cv2.Canny(binary_img, 20, 240)
# 直线检测
def line_detection(edges):
    return cv2.HoughLinesP(edges, 1, np.pi/180, 100, maxLineGap=25)
# 计算斜率
def k_of_lines(lines):
    k = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        k.append((y1 - y2) / (x1 - x2))
    return k
def largestConnectComponent(bw_img):
    '''
    compute largest Connect component of an labeled image

    Parameters:
    ---

    bw_img:
        binary image

    Example:
    ---
        >>> lcc = largestConnectComponent(bw_img)

    '''

    labeled_img, num = label(bw_img, neighbors=4, background=0, return_num=True)    
    # plt.figure(), plt.imshow(labeled_img, 'gray')

    max_label = 0
    max_num = 0
    for i in range(1, num): # 这里从1开始，防止将背景设置为最大连通域
        if np.sum(labeled_img == i) > max_num:
            max_num = np.sum(labeled_img == i)
            max_label = i
    lcc = (labeled_img == max_label)

    return lcc
# 得到的平行线可能是同一侧的，判断算法：
#   - 计算端点两距离得到四个值
#   - if 极差超出阈值说明线段靠近
def judge_line_distance (l1, l2):
    xa, ya, xb, yb = l1[0]
    xc, yc, xd, yd = l1[0]
    d1 = cal_line_len(xa, ya, xc, yc)
    d2 = cal_line_len(xa, ya, xd, yd)
    d3 = cal_line_len(xb, yb, xc, yc)
    d4 = cal_line_len(xb, yb, xd, yd)
    print(max(d1, d2, d3, d4) - min(d1, d2, d3, d4))
    return max(d1, d2, d3, d4) - min(d1, d2, d3, d4) <= CLOSER_PARALLEL_LINE_THRESH
# 找图像收信

gray_image = cvt_gray(img)
dump_process_img('gray', gray_image)

clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
dst = clahe.apply(gray_image)
dump_process_img('equ', dst)

# #线性变换
# a = 2
# O = float(a) * gray_image
# O[O>255] = 255 #大于255要截断为255
# #数据类型的转换
# O = np.round(O)
# O = O.astype(np.uint8)
# dump_process_img('equ', O)


binary_img = cvt_binary(gray_image)
dump_process_img('row_binary', binary_img)

binary_img = blur(binary_img)
dump_process_img('blured_binary', binary_img)

#创建矩形结构单元
g = cv2.getStructuringElement(cv2.MORPH_RECT,(9,9))
#形态学处理,开运算
img_open = cv2.morphologyEx(binary_img,cv2.MORPH_OPEN,g)
dump_process_img('open', img_open)

edges = edge_detection(binary_img)
dump_process_img('edges', edges)

lines = line_detection(edges)
print_lines(np.zeros(edges.shape), 'lines',lines)

k = k_of_lines(lines)

# 裁剪比例尺区域
k_0_lines = []
for i in range(len(k)):
    if abs(k[i]) <= 0.05:
        k_0_lines.append(lines[i])

x_min = 999999999
for line in k_0_lines:
    x1, y1, x2, y2 = line[0]
    if min(x1, x2) < x_min:
        x_min = min(x1, x2)
lines_idx_to_del = []
# 排除规则：
#   - 
for i in range(len(lines)):
    x1, y1, x2, y2 = lines[i][0]
    mid_height = edges.shape[0] * 0.66
    if x1 >= x_min or \
        x2 >= x_min or \
        y1 <= mid_height or \
        y2 <= mid_height or \
        x1 == x2 or \
        K_THRESH[0] < k[i] < K_THRESH[1] or \
        cal_line_len(x1, y1, x2, y2) < LINE_LEN_THRESH:
        lines_idx_to_del.append(i)
lines = np.delete(lines, lines_idx_to_del, axis=0)
k = np.delete(k, lines_idx_to_del)
print_lines(np.zeros(edges.shape), 'filtered_lines',lines)

parallel_lines_idx = []
y_max = edges.shape[1]
for k1_idx in range(len(k)):
    for k2_idx in range(k1_idx + 1, len(k)):
        if abs(k[k1_idx] - k[k2_idx]) <= PARALLEL_THRESH:
            parallel_lines_idx.append((k1_idx, k2_idx, abs(k[k1_idx] - k[k2_idx])))


# 先随便选一个，后面优化再处理，可能出现选中相邻的两条线
chosen_line_pair = random.choice(parallel_lines_idx)
# min_k_diff = 0
# if len(parallel_lines_idx) > 1:
#     for pl in parallel_lines_idx:
    #     l1 = lines[pl[0]]
    #     l2 = lines[pl[1]]
    #     if judge_line_distance(l1, l2):
    #         chosen_line_pair = pl


# 绘制平行线判定结果
# empty = np.zeros(edges.shape)
# l1 = lines[chosen_line_pair[0]]
# l2 = lines[chosen_line_pair[1]]
# x1, y1, x2, y2 = l1[0]
# cv2.line(empty, (x1, y1), (x2, y2), 255, 2)
# x1, y1, x2, y2 = l2[0]
# cv2.line(empty, (x1, y1), (x2, y2), 255, 2)
# dump_process_img('test', empty)

# 计算手腕中线，一会再算

# 旋转
(h, w) = edges.shape[:2]
p = (w // 2, h // 2)

angle = math.atan(k[chosen_line_pair[0]]) * 180 / math.pi + 90
M = cv2.getRotationMatrix2D(p, angle, 1.0)
rotated = cv2.warpAffine(edges[:, :x_min], M, (w, h))
rotated_raw = cv2.warpAffine(img, M, (w, h))
dump_process_img('rotated', rotated)
dump_process_img('rotated-row', rotated_raw)

M = cv2.moments(rotated)
cX = int(M["m10"] / M["m00"])
cY = int(M["m01"] / M["m00"])
print(cX, cY)
cv2.circle(rotated_raw, (cX, cY), 5, (255, 0, 0), -1)
dump_process_img('mid-point', rotated_raw)




