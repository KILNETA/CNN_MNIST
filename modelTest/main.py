import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model

model = load_model('mnist_cnn.h5')  # 选取自己的.h模型名称
plt.figure()

for i in range(10):
    image = cv2.imread(str(i)+'a.png')
    img = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY) # RGB图像转为gray

    #需要用reshape定义出例子的个数，图片的 通道数，图片的长与宽。具体的参加keras文档
    img = (img.reshape(1, 28, 28, 1)).astype('int32')/255
    predict = model.predict(img)
    classes_x=np.argmax(predict,axis=1)

    probs = model.predict(img, batch_size=1)

    print (str(i)+'a 识别为：'+ str(classes_x))

    # 繪出圖表的預測結果
    plt.subplot(1, 2, 1)
    plt.title("Example of Digit:" + str(classes_x))
    plt.imshow(image, cmap="gray")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.title("Probabilities for Each Digit Class")
    plt.bar(np.arange(10), probs.reshape(10), align="center")
    plt.xticks(np.arange(10), np.arange(10).astype(str))
    plt.show()