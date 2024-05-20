import markovify
import sys

# Read text from file
if len(sys.argv) != 2:
    sys.exit("Usage: python generator.py sample.txt")
with open(sys.argv[1]) as f:
    text = f.read()

# Train model
# 使用读取的文本训练一个马尔可夫链模型。
text_model = markovify.Text(text)

# Generate sentences
# 生成5个随机句子并打印出来。text_model.make_sentence()方法基于训练模型生成一个随机句子。
print()
for i in range(5):
    print(text_model.make_sentence())
    print()
