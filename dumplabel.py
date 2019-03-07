import cv2
import numpy as np
import pickle
import sys
import os

data_dir_path = u"../data/test"
file_list = os.listdir(r'../data/test')
target_all = []
imgn = 0
buildn = 0

for file_name in file_list:
    root, ext = os.path.splitext(file_name)

    if ext == u'.tif':
        # target画像の読み込み
        abs_name = data_dir_path + '/' + file_name
        img = cv2.imread(abs_name) 
        print("Load {}".format(abs_name))

        # グレースケール化
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 大津の二値化
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # ラベリング処理
        label = cv2.connectedComponentsWithStats(gray)

        # オブジェクト情報を項目別に抽出
        n = label[0] - 1
        data = np.delete(label[2], 0, 0)
        center = np.delete(label[3], 0, 0)

        target_bbox = np.empty((0, 4), dtype=np.float32)
        target_label = np.empty(0, dtype=np.int32)

        for i in range(n):
            #print(i)
            x0 = float(data[i][0])
            y0 = float(data[i][1])
            x1 = float(data[i][0] + data[i][2])
            y1 = float(data[i][1] + data[i][3])
            target_bbox = np.append(target_bbox, np.array([[y0, x0, y1, x1]]), axis=0)
            target_label = np.append(target_label, 0)
 
        #abs_name2 = abs_name + 'f'
        #target_all.append((os.path.join(abs_name2), target_bbox, target_label))
        img2 = cv2.imread(abs_name + 'f')
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        img2 = np.array(img2, dtype=np.float32)
        img2 = np.transpose(img2, (2, 0, 1))
        
        if n != 0:
            target_all.append((img2, target_bbox, target_label))
            imgn = imgn + 1
            buildn = buildn + n

#print(target_all)
print("imgn=", imgn)
print("buildn=", buildn)

with open('test-image-label.pkl', 'wb') as wf:
    pickle.dump(target_all, wf)

with open('test-image-label.pkl', 'rb') as rf:
    image_label = pickle.load(rf)
