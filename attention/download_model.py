# download_model.py

from huggingface_hub import snapshot_download
import os
import shutil

def download_bert_model(model_name="bert-base-uncased", save_path="D://bert"):
    """
    下载 BERT 模型和分词器，并将其保存到本地。
    """
    # 下载模型和分词器的快照
    print(f"Downloading {model_name} from Hugging Face Hub...")
    model_dir = snapshot_download(repo_id=model_name)

    # 将下载的内容复制到指定的保存路径
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    shutil.copytree(model_dir, save_path)
    print(f"Model and tokenizer saved locally in {save_path}")

if __name__ == "__main__":
    download_bert_model()

