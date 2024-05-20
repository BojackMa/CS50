from PIL import Image

# 打开PPM文件
image = Image.open('gtsrb-small/0/00014_00007.ppm')

# 显示图片
image.show()