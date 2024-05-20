import cv2
import numpy as np
import os
import sys
import tensorflow as tf


from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.
    load_data 函数应该接受一个参数 data_dir，代表存储数据的目录路径，并返回数据集中每张图片的图像数组和标签。
    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    images = []
    labels = []
    # 遍历每个数字编号的类别目录
    for category in range(NUM_CATEGORIES):
        # 生成当前文件路径
        category_path = os.path.join(data_dir, str(category))
        # 确保目录存在
        if not os.path.exists(category_path):
            continue

        # 遍历类别目录中的所有图像文件
        for filename in os.listdir(category_path):
            img_path = os.path.join(category_path, filename)

            # 读取图像并调整其大小
            img = cv2.imread(img_path)
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

            # 将图像数据添加到图像列表中
            images.append(img)
            # 将图像的类别标签添加到标签列表中
            labels.append(category)

    return (images, labels)


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    model = tf.keras.models.Sequential([
        # 卷积层，使用ReLU激活函数
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        # 池化层
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        # 防止过拟合
        tf.keras.layers.Dropout(0.5),
        # 更多的卷积层和池化层可以根据需要添加
        # 32变成64的原因是随着网络深度的增加，图像被抽象成更高层次的特征（如从简单的边缘到纹理再到对象的部分）。在这一过程中，可能需要更多的滤波器来捕捉更复杂和更细微的特征。增加滤波器的数量可以帮助网络学习更多的特征表示，这对于理解和分类图像尤其重要。
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.5),
        # 将特征图展平
        tf.keras.layers.Flatten(),
        # 全连接层
        tf.keras.layers.Dense(500, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        # 输出层，使用softmax激活函数
        tf.keras.layers.Dense(NUM_CATEGORIES, activation='softmax')
    ])

    # 编译模型
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model



if __name__ == "__main__":
    main()
