from PIL import Image
import os

images = os.listdir("../data/test") #ディレクトリのパス

for i in images:
        filename = "../data/test/" + i
        img = Image.open(filename)
        print("Load {}".format(filename))
        img = img.resize((750, 750))
        img.save(filename) #上書き保存
