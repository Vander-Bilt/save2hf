import requests
import json

def upload_image_to_imgbb(api_key, file_path):
    """
    将本地图片上传到ImgBB。

    参数:
    api_key (str): 你的ImgBB API密钥。
    file_path (str): 本地图片文件的完整路径。

    返回:
    dict: 包含上传结果的字典，如果上传失败则返回None。
    """
    url = "https://api.imgbb.com/1/upload"
    
    try:
        # 以二进制模式打开图片文件
        with open(file_path, "rb") as file:
            # 准备请求参数和文件
            payload = {
                "key": api_key,
            }
            # files参数会自动处理multipart/form-data
            files = {
                "image": file,
            }

            print("正在上传图片...")
            response = requests.post(url, data=payload, files=files)
            
            # 检查响应状态码
            if response.status_code == 200:
                result = response.json()
                if result['success']:
                    print("图片上传成功！")
                    return result['data']
                else:
                    print(f"图片上传失败: {result['error']['message']}")
                    return None
            else:
                print(f"请求失败，状态码：{response.status_code}")
                print(f"响应内容：{response.text}")
                return None
    except FileNotFoundError:
        print(f"错误：文件未找到，请检查路径是否正确: {file_path}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"网络请求错误: {e}")
        return None

# --- 使用示例 ---
if __name__ == "__main__":
    # 替换成你的API密钥
    your_api_key = "0f342c8941a83e596d25cd51a11efd6a"
    # 替换成你的图片文件路径
    your_image_path = "/Users/dinaqian/Downloads/xyj.jpeg"
    
    upload_data = upload_image_to_imgbb(your_api_key, your_image_path)
    
    if upload_data:
        print("\n--- 上传结果 ---")
        print(f"图片URL: {upload_data['url']}")
        print(f"缩略图URL: {upload_data['thumb']['url']}")
        print(f"删除URL: {upload_data['delete_url']}")
    else:
        print("\n上传流程未成功完成。")