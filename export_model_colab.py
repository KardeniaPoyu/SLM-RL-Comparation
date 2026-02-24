import shutil
import os
import argparse

try:
    from google.colab import files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
    print("⚠️ 警告: 未在 Google Colab 环境中运行，打包后将不会自动触发浏览器下载。")

def export_model(model_dir, zip_name):
    """打包模型并触发下载"""
    if os.path.exists(model_dir):
        print(f"✅ 找到模型路径: {model_dir}")
        # 打包成 zip
        shutil.make_archive(zip_name, 'zip', model_dir)
        print(f"📦 压缩完成: {zip_name}.zip")
        
        # 如果在 Colab 中，则触发下载
        if IN_COLAB:
            print("🚀 正在弹出浏览器下载窗口...")
            files.download(f"{zip_name}.zip")
        else:
            print(f"💡 可在当前目录找到压缩包：{zip_name}.zip")
    else:
        print(f"❌ 路径不存在，请确认模型文件夹 {model_dir} 是否存在。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="打包并下载训练好的模型")
    parser.add_argument("--model_dir", type=str, default="saved_models/grpo_update_40", help="要打包的模型所在文件夹的路径")
    parser.add_argument("--zip_name", type=str, default="GRPO_40", help="生成的压缩包名称（不带.zip）")
    
    args = parser.parse_args()
    
    export_model(args.model_dir, args.zip_name)
    
