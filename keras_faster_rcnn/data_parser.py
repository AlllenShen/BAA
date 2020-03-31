import os
import random
import math
import json
import pprint

def get_data():
    '''
    将label数据转换成和demo一样的格式
    {
        width:
        height:
        bboxes: [
            {x1: x2: y1: y2: class: }
        ]
        imageset:
        filepath
    }
    '''
    DATA_PATH = os.path.abspath('data')
    LABEL_PATH = os.path.join(DATA_PATH, 'labels')
    IMG_PATH = os.path.join(DATA_PATH, 'raw')
    TRAIN_SPLIT = 0.6
    VAL_SPLIT = 0.2

    classes_count = {}  # 一个字典，key为对应类别名称，value对应为类别所对应的样本（标注框）个数
    classes_mapping = {}  # 一个字典数据结构，key为对应类别名称，value为对应类别的一个标识index
    img_data = []

    label_files = os.listdir(LABEL_PATH)
    train_files = random.sample(
        label_files, math.ceil(len(label_files) * TRAIN_SPLIT))
    temp = [file for file in label_files if file not in train_files]
    val_files = random.sample(temp, math.ceil(len(label_files) * VAL_SPLIT))
    test_files = [file for file in temp if file not in val_files]
    # print('train: {}, val: {}, test: {}'.format(
    #     len(train_files), len(val_files), len(test_files)))

    def handle_json(file, type):
        data = json.load(open(file))
        bboxes = []
        for shape in data['shapes']:
            classes_name = shape['label']
            bboxes.append({
                'x1': shape['points'][0][0],
                'y1': shape['points'][0][1],
                'x2': shape['points'][1][0],
                'y2': shape['points'][1][1],
                'class': classes_name
            })
            if classes_name in classes_count:  # classes_count 存储类别以及对应类别的标注框个数
                classes_count[classes_name] += 1
            else:
                classes_count[classes_name] = 1
            if classes_name not in classes_mapping:
                classes_mapping[classes_name] = len(classes_mapping)
        img_name = file.split(os.sep)[-1].replace('json', 'jpg')
        return {
            'width': data['imageWidth'],
            'height': data['imageHeight'],
            'bboxes': bboxes,
            'filepath': os.path.join(IMG_PATH, img_name),
            'imageset': type
        }
    for file in train_files:
        img_data.append(handle_json(os.path.join(LABEL_PATH, file), 'train'))
    for file in val_files:
        img_data.append(handle_json(os.path.join(LABEL_PATH, file), 'val'))
    for file in test_files:
        img_data.append(handle_json(os.path.join(LABEL_PATH, file), 'test'))
    return img_data, classes_count, classes_mapping


if __name__ == "__main__":
    image_data, classes_count, classes_mapping = get_data()
    print("数据集大小：", len(image_data))
    print("类别个数：", len(classes_count))
    print("类别种类：", classes_count.keys())
    print("打印其中一条样本数据：")
    pprint.pprint(image_data[0])
