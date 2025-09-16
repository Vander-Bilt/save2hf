import io
import os
import requests
import hashlib
import numpy as np
from PIL import Image


def dencrypt_image_v2(image:Image.Image, psw): 
    width = image.width 
    height = image.height 
    x_arr = [i for i in range(width)] 
    shuffle_arr(x_arr,psw) 
    y_arr = [i for i in range(height)] 
    shuffle_arr(y_arr,get_sha256(psw)) 
    pixel_array = np.array(image) 


    pixel_array = np.transpose(pixel_array, axes=(1, 0, 2)) 
    for x in range(width-1,-1,-1): 
        _x = x_arr[x] 
        temp = pixel_array[x].copy() 
        pixel_array[x] = pixel_array[_x] 
        pixel_array[_x] = temp 
    pixel_array = np.transpose(pixel_array, axes=(1, 0, 2)) 
    for y in range(height-1,-1,-1): 
        _y = y_arr[y] 
        temp = pixel_array[y].copy() 
        pixel_array[y] = pixel_array[_y] 
        pixel_array[_y] = temp 


    image.paste(Image.fromarray(pixel_array)) 
    return image 


def get_range(input:str,offset:int,range_len=4): 
    offset = offset % len(input) 
    return (input*2)[offset:offset+range_len] 

def get_sha256(input:str): 
    hash_object = hashlib.sha256() 
    hash_object.update(input.encode('utf-8')) 
    return hash_object.hexdigest() 

def shuffle_arr(arr,key): 
    sha_key = get_sha256(key) 
    key_len = len(sha_key) 
    arr_len = len(arr) 
    key_offset = 0 
    for i in range(arr_len): 
        to_index = int(get_range(sha_key,key_offset,range_len=8),16) % (arr_len -i) 
        key_offset += 1 
        if key_offset >= key_len: key_offset = 0 
        arr[i],arr[to_index] = arr[to_index],arr[i] 
    return arr 


def upload(imgbb_api_key, filepaths, password):
    """
    将本地图片上传到ImgBB。
    参数:
    api_key (str): 你的ImgBB API密钥。
    file_path (str): 本地图片文件的完整路径。
    返回:
    dict: 包含上传结果的字典，如果上传失败则返回None。
    """
    url = "https://api.imgbb.com/1/upload"
    if not filepaths:
        return ("No files to upload.",)
    
    try:
        output_paths = []
        # output_thumb_paths = []
        print(f"filepaths got: {filepaths}")
        for file_path in filepaths:
            if not isinstance(file_path, str) or not os.path.exists(file_path):
                print(f"File not found or invalid path, skipping: {file_path}")
                continue
            print(f"file_path: {file_path}")
            img = Image.open(file_path)
            decrypted_img = dencrypt_image_v2(img, get_sha256(password))
            
            # 使用 BytesIO 在内存中保存图像数据
            img_byte_arr = io.BytesIO()
            decrypted_img.save(img_byte_arr, format='PNG')  # 或者 'JPEG'，根据需要选择
            img_byte_arr.seek(0)  # 将指针移回文件开头
            # 准备请求参数和文件
            payload = {
                "key": imgbb_api_key,
            }
            # files参数会自动处理multipart/form-data
            files = {
                "image": (os.path.basename(file_path), img_byte_arr, 'image/png'),
            }
            print("正在上传图片...")
            response = requests.post(url, data=payload, files=files)
            
            # 检查响应状态码
            if response.status_code == 200:
                result = response.json()
                if result['success']:
                    print(f"图片 {file_path} 上传成功！")
                    upload_data = result['data']
                    print(f"upload_data: {upload_data}")
                    # print(upload_data['url'])
                    # print(upload_data['thumb']['url'])
                    output_paths.append(f"{upload_data['url']}|||{upload_data['thumb']['url']}")
                    # output_thumb_paths.append(upload_data['thumb']['url'])
                else:
                    print(f"图片上传失败: {result['error']['message']}")
            else:
                print(f"请求失败，状态码：{response.status_code}")
                print(f"响应内容：{response.text}")
        
        return (",".join(output_paths),)
    except Exception as e:
        return (f"Upload failed: {str(e)}",)

your_api_key = "0f342c8941a83e596d25cd51a11efd6a"
# 替换成你的图片文件路径
your_image_path = ["/Users/dinaqian/Downloads/ComfyUI_000011.png"]

upload(your_api_key, your_image_path, "Bilt8")
