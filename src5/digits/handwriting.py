import sys
import tensorflow as tf

# Use MNIST handwriting dataset
mnist = tf.keras.datasets.mnist

# Prepare data for training
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0   # 对图像数据进行归一化处理，即将像素值从 [0, 255] 缩放到 [0, 1] 范围，这有助于模型的训练效率和收敛速度。
y_train = tf.keras.utils.to_categorical(y_train)    # 标签（y_train 和 y_test）被转换为独热编码（one-hot encoding），这是分类问题常用的标签表示方式。
y_test = tf.keras.utils.to_categorical(y_test)
# 最后，输入数据被重塑为四维数组以适配 TensorFlow 处理图像的需求，其中最后一个维度 1 表示图像的颜色通道数（灰度图）。
# 在深度学习中，特别是处理分类问题时，通常需要将类别标签转换为一种更适合神经网络处理的格式。这个格式通常是独热编码（One-Hot Encoding），tf.keras.utils.to_categorical函数就是用来完成这种转换的。
# 具体来说，to_categorical 函数将整数类别标签转换为二进制的独热编码形式。在独热编码中，类别标签被转换为一个向量，这个向量的长度等于类别的总数，向量中的所有元素都是0，除了表示该类别的索引位置是1。
# y_train = [0, 2, 1]
# 输出的结果是
# [[1., 0., 0.],
#  [0., 0., 1.],
#  [0., 1., 0.]]
# 这种转换非常重要，因为大多数神经网络的输出层通常使用 softmax 激活函数，该函数可以输出一个概率分布，最适合与独热编码的格式相匹配。这样可以直接使用类别交叉熵（categorical crossentropy）作为损失函数，从而评估模型的预测与真实标签之间的差异。
# 在处理图像数据时，尤其是使用卷积神经网络（CNN）时，通常需要将图像数据的维度明确设置为 (样本数, 高度, 宽度, 通道数) 的格式。这个格式让模型能够正确处理每个图像样本的空间维度（高度和宽度）和通道数（例如，彩色图像的RGB三通道或灰度图像的单通道）。
# 例如，如果 x_train 最初的形状是 (60000, 28, 28)（表示有 60000 张 28x28 的灰度图像），重塑后的形状将会是 (60000, 28, 28, 1)。这样的数据格式调整是为了让网络能够正确识别和处理每个独立的图像样本，同时保持图像原有的空间结构信息。
x_train = x_train.reshape(
    x_train.shape[0], x_train.shape[1], x_train.shape[2], 1
)
x_test = x_test.reshape(
    x_test.shape[0], x_test.shape[1], x_test.shape[2], 1
)

# Create a convolutional neural network
# 构建卷积神经网络,构建了一个顺序模型（Sequential）

# 其实原理很简单：卷积（很强大的特征提取），压缩泛化（池化），喂给全连接层（需要展平）
# 包括多个全连接层（Dense层）可以提高模型的学习能力和表示能力。每个全连接层都能从前一层学习到的特征中进一步抽取信息，加深网络的可以帮助解决更复杂的问题。在这个特定实验中，使用了两个全连接层
# 全连接层能够将学习到的特征转化为更高层次的抽象表示。在多层全连接网络中，每一层可以基于前一层的输出构建更复杂的特征表示。例如，第一层可能识别简单的模式和结构，而第二层则可能组合这些模式以识别更复杂的概念。
model = tf.keras.models.Sequential([

    # Convolutional layer. Learn 32 filters using a 3x3 kernel
    tf.keras.layers.Conv2D(
        32, (3, 3), activation="relu", input_shape=(28, 28, 1)
    ),
    # activation: 指定卷积层后使用的激活函数。这里使用的是ReLU（Rectified Linear Unit），它的公式是 f(x) = max(0, x)。ReLU函数有助于在网络中引入非线性因素，而非线性是解决复杂问题的关键，同时ReLU也有助于减少梯度消失问题，加快网络的收敛速度。
    # input_shape: 这是模型第一层特有的参数，用于定义输入数据的形状。这里 (28, 28, 1) 表示每个输入的图像是28x28像素，且是单通道的（灰度图）。这个参数仅在模型的第一层设置，因为后续层的输入形状可以自动计算。

    # Max-pooling layer, using 2x2 pool size
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    # 在卷积神经网络中，特征通过卷积层和池化层被提取和压缩后，通常会有一个或多个全连接层（也称为密集层）来进一步处理这些特征，以进行分类或其他任务。
    # Flatten层的主要作用是将前面卷积层或池化层输出的多维特征图（feature map）转换为一维数组。这是因为全连接层（Dense层）需要一维的输入向量。
    # 展平操作并不涉及任何乘法或数学运算，而仅仅是改变数据的形状！！！将多维数据“拉直”为一维！！！
    # Flatten units
    tf.keras.layers.Flatten(),

    # 全连接层，也称为密集层（Dense层），是神经网络中最基本的一种层，其中每个输入节点都与输出层的每个节点通过权重连接。这意味着网络中任何一个输入特征都可以影响输出层的每一个节点，使得这种层非常适合从整体输入特征中学习模式。
    # 全连接层的工作原理是对输入特征进行加权和，然后通常再加上一个偏置项，最后通过一个激活函数。
    # output=activation(W×input+b)  W 是权重矩阵[n,128]，input 是输入向量[1,n]，b是偏置向量，activation 是激活函数
    # Add a hidden layer with dropout
    # 这里定义了128个神经元，这意味着该层会有128个输出，更多的神经元可以使网络有更大的表示能力，即能学习更复杂的函数，在实践中，神经元的数量经常是基于实验调整得到的。
    tf.keras.layers.Dense(128, activation="relu"),
    # Dropout层在训练过程中随机地“丢弃”（即设置为零）层的输出特征的一部分。这里的0.5参数意味着有50%的几率一个神经元的输出在训练时会被置零。这是一种正则化技术，用于防止神经网络过拟合。通过随机丢弃一部分网络单元，它迫使网络学习更加鲁棒的特征，这些特征在网络的多个不同随机子集中都有效，从而提高了模型的泛化能力。
    tf.keras.layers.Dropout(0.5),
    # Dense Layer：这是输出层，同样是一个全连接层。这里定义的输出单元数（10个）对应于分类任务的类别数目，即MNIST数据集中的10个数字（0到9）。
    # Activation Function：使用了softmax激活函数，它可以将输出解释为概率分布，每个输出单元的值在0到1之间，且所有单元的总和为1。在多分类任务中，这允许模型输出每个类别的预测概率。
    # Add an output layer with output units for all 10 digits
    tf.keras.layers.Dense(10, activation="softmax")
])

# Train neural network
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
model.fit(x_train, y_train, epochs=10)

# Evaluate neural network performance
model.evaluate(x_test,  y_test, verbose=2)

# Save model to file
if len(sys.argv) == 2:
    filename = sys.argv[1]
    model.save(filename)
    print(f"Model saved to {filename}.")

