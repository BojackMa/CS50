import sys
import tensorflow as tf

from PIL import Image, ImageDraw, ImageFont
from transformers import AutoTokenizer, TFBertForMaskedLM

# Pre-trained masked language model
MODEL = "D://bert"

# Number of predictions to generate
K = 3

# Constants for generating attention diagrams 定义生成注意力图表所需的常量，包括字体、网格大小和每个单词的像素数。
FONT = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 28)
GRID_SIZE = 40
PIXELS_PER_WORD = 200


def main():
    text = input("Text: ")

    # Tokenize input
    # 使用预训练模型初始化分词器，将输入文本分词，并查找掩码标记的索引。如果找不到掩码标记，则退出程序并提示错误信息。
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    inputs = tokenizer(text, return_tensors="tf")
    mask_token_index = get_mask_token_index(tokenizer.mask_token_id, inputs)
    if mask_token_index is None:
        sys.exit(f"Input must include mask token {tokenizer.mask_token}.")

    # Use model to process input
    # 使用预训练模型初始化BERT，将分词后的输入传递给模型，并获取输出，包括注意力分数。
    model = TFBertForMaskedLM.from_pretrained(MODEL)
    result = model(**inputs, output_attentions=True)
    # result 对象包含以下信息
    # logits: 模型的输出logits，通常用于分类或语言建模任务。它包含了每个输入标记对应的词汇表中每个词的预测分数（logits）。
    # attentions: 注意力分数，表示每个注意力头在处理输入时关注每个标记的程度。
    # hidden_states: 每一层的隐藏状态（如果 output_hidden_states=True 被设置）。


    # Generate predictions
    # 生成预测，找到掩码标记位置的输出logits，选择前K个最可能的标记，并将掩码标记替换为预测的标记。
    # logits用于表示每个词汇作为掩码标记位置预测结果的得分
    mask_token_logits = result.logits[0, mask_token_index]
    # 返回的是词汇表中所有词在掩码标记位置的预测分数。具体来说，这些分数表示模型在掩码标记位置上对每个词汇的预测概率（通常是在logits尺度上的分数，未经过归一化）。
    top_tokens = tf.math.top_k(mask_token_logits, K).indices.numpy()
    for token in top_tokens:
        print(text.replace(tokenizer.mask_token, tokenizer.decode([token])))

    # Visualize attentions
    # 调用 visualize_attentions 函数生成注意力图表。
    visualize_attentions(inputs.tokens(), result.attentions)


def get_mask_token_index(mask_token_id, inputs):
    """
    Return the index of the token with the specified `mask_token_id`, or
    `None` if not present in the `inputs`.
    """

    # 获取 input_ids，形状为 (batch_size, sequence_length)
    input_ids = inputs["input_ids"].numpy()

    # 遍历 input_ids 查找 mask_token_id
    for batch_index in range(input_ids.shape[0]):
        for token_index in range(input_ids.shape[1]):
            if input_ids[batch_index][token_index] == mask_token_id:
                return token_index  # 返回 token_index，而不是 batch_index

    # 如果没有找到，返回 None
    return None



def get_color_for_attention_score(attention_score):
    """
    Return a tuple of three integers representing a shade of gray for the
    given `attention_score`. Each value should be in the range [0, 255].
    """
    # 确保 attention_score 在 0 到 1 之间
    assert 0.0 <= attention_score <= 1.0, "Attention score must be between 0 and 1"

    # 将 attention_score 映射到 0 到 255 的范围
    gray_value = int(attention_score * 255)

    # 返回灰度值的 RGB 三元组
    return (gray_value, gray_value, gray_value)



def visualize_attentions(tokens, attentions):
    """
    Produce a graphical representation of self-attention scores.

    For each attention layer, one diagram should be generated for each
    attention head in the layer. Each diagram should include the list of
    `tokens` in the sentence. The filename for each diagram should
    include both the layer number (starting count from 1) and head number
    (starting count from 1).
    """
    # tokens是这玩意 tokens = ['[CLS]', 'the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', '[MASK]', '.', '[SEP]']
    # TODO: Update this function to produce diagrams for all layers and heads.
    # 遍历所有注意力层
    for layer_index, layer_attentions in enumerate(attentions):
        # 遍历当前层中的每个注意力头
        for head_index, head_attentions in enumerate(layer_attentions[0]):
            # 生成注意力图表
            generate_diagram(
                layer_index + 1,
                head_index + 1,
                tokens,
                head_attentions
            )


def generate_diagram(layer_number, head_number, tokens, attention_weights):
    """
    Generate a diagram representing the self-attention scores for a single
    attention head. The diagram shows one row and column for each of the
    `tokens`, and cells are shaded based on `attention_weights`, with lighter
    cells corresponding to higher attention scores.

    The diagram is saved with a filename that includes both the `layer_number`
    and `head_number`.
    """
    # Create new image
    image_size = GRID_SIZE * len(tokens) + PIXELS_PER_WORD
    img = Image.new("RGBA", (image_size, image_size), "black")
    draw = ImageDraw.Draw(img)

    # Draw each token onto the image
    for i, token in enumerate(tokens):
        # Draw token columns
        token_image = Image.new("RGBA", (image_size, image_size), (0, 0, 0, 0))
        token_draw = ImageDraw.Draw(token_image)
        token_draw.text(
            (image_size - PIXELS_PER_WORD, PIXELS_PER_WORD + i * GRID_SIZE),
            token,
            fill="white",
            font=FONT
        )
        token_image = token_image.rotate(90)
        img.paste(token_image, mask=token_image)

        # Draw token rows
        _, _, width, _ = draw.textbbox((0, 0), token, font=FONT)
        draw.text(
            (PIXELS_PER_WORD - width, PIXELS_PER_WORD + i * GRID_SIZE),
            token,
            fill="white",
            font=FONT
        )

    # Draw each word
    for i in range(len(tokens)):
        y = PIXELS_PER_WORD + i * GRID_SIZE
        for j in range(len(tokens)):
            x = PIXELS_PER_WORD + j * GRID_SIZE
            color = get_color_for_attention_score(attention_weights[i][j])
            draw.rectangle((x, y, x + GRID_SIZE, y + GRID_SIZE), fill=color)

    # Save image
    img.save(f"Attention_Layer{layer_number}_Head{head_number}.png")


if __name__ == "__main__":
    main()
