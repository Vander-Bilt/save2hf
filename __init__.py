import io
import os
import torch
import opennsfw2 as n2
from huggingface_hub import HfApi, HfFolder
import folder_paths
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
import zlib
import base64
import requests
import hashlib
import numpy as np
from PIL import Image as PILImage # Use an alias to avoid conflict with your patched class

# Define the NSFW probability threshold
# MAX_PROBABILITY = 0.65

class PushToHFDataset:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hf_token": ("STRING", {"default": ""}),
                "dataset_name": ("STRING", {"default": ""}),
                "huggingface_path_in_repo": ("STRING", {"default": ""}),
                "filepaths": ("STRING[]", {}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("result",)
    FUNCTION = "push"
    CATEGORY = "utils"


    def push(self, hf_token, dataset_name, huggingface_path_in_repo, filepaths):
        api = HfApi()
        HfFolder.save_token(hf_token)
        
        print(f"filepaths got: {filepaths}")

        if not filepaths:
            return ("No files to upload.",)
        
        try:
            output_paths = []
            for file_path in filepaths:
                if not isinstance(file_path, str) or not os.path.exists(file_path):
                    print(f"File not found or invalid path, skipping: {file_path}")
                    continue

                path_in_repo = os.path.join(huggingface_path_in_repo, os.path.basename(file_path))
                output_paths.append(path_in_repo)

                api.upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=path_in_repo,
                    repo_id=dataset_name,
                    repo_type="dataset",
                    token=hf_token,
                )

            # return (f"Uploaded {len(filepaths)} files to {dataset_name}.",)
            return (",".join(output_paths),)
        except Exception as e:
            return (f"Upload failed: {str(e)}",)


class NSFWFilter:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The image(s) to check and filter."}),
                "enabled": ("BOOLEAN", {"default": True, "tooltip": "Whether to enable the NSFW filter."}),
                "PROBABILITY": ("FLOAT", {"default": 0.65, "tooltip": "NSFW probability threshold."}),
            },
        }

    RETURN_TYPES = ("IMAGE","FLOAT",)
    RETURN_NAMES = ("filtered_images","nsfw_probabilities",)
    FUNCTION = "filter_images"
    CATEGORY = "utils"
    DESCRIPTION = "Filters images based on NSFW probability. Replaces high-risk images with a blank image."

    def filter_images(self, images, enabled, PROBABILITY):
        filtered_images = []
        nsfw_probs = []

        batch_size, height, width, channels = images.shape
        blank_image = torch.zeros((1, height, width, channels), dtype=images.dtype, device=images.device)

        # Process each image in the batch
        for image_tensor in images:
            i = 255. * image_tensor.cpu().numpy()
            original_img = PILImage.fromarray(np.clip(i, 0, 255).astype(np.uint8)).convert('RGB')

            nsfw_prob = 0.0
            try:
                nsfw_prob = n2.predict_image(original_img)

            except Exception as e:
                print(f"Error during NSFW detection: {e}. Defaulting probability to 0.0")
            
            

            # 核心逻辑: 根据 enabled 状态和概率决定输出
            if enabled:
                # 如果启用过滤，并且概率超过阈值，则替换为黑图
                if nsfw_prob > PROBABILITY:
                    print(f"NSFW filter is ENABLED. Probability ({nsfw_prob:.4f}) is above threshold ({PROBABILITY}). RESET it to 0. Replacing with blank image.")
                    filtered_images.append(blank_image)
                    nsfw_prob = 0
                # 否则，保留原图
                else:
                    print(f"NSFW filter is ENABLED. Probability ({nsfw_prob:.4f}) is acceptable. Keeping original image.")
                    filtered_images.append(image_tensor.unsqueeze(0))
            else:
                # 如果未启用过滤，则始终保留原图
                print(f"NSFW filter is DISABLED. Probability detected: {nsfw_prob:.4f}. Passing through original image.")
                filtered_images.append(image_tensor.unsqueeze(0))

            nsfw_probs.append(nsfw_prob)

        # Concatenate the list of processed tensors back into a single batch tensor
        return_images = torch.cat(filtered_images, dim=0)

        # 返回结果：根据 enabled 状态，返回过滤后的图片和概率
        return (return_images, nsfw_probs)

class PushToImageBB:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "imgbb_api_key": ("STRING", {"default": ""}),
                "filepaths": ("STRING[]", {}),
                "nsfw_probabilities": ("FLOAT", {}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_paths",)
    FUNCTION = "upload"
    CATEGORY = "utils"

    def upload(self, imgbb_api_key, filepaths, nsfw_probabilities):
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
            # print(f"filepaths got: {filepaths}")

            idx = 0
            for file_path in filepaths:
                if not isinstance(file_path, str) or not os.path.exists(file_path):
                    print(f"File not found or invalid path, skipping: {file_path}")
                    continue

                nsfw_prob = nsfw_probabilities[idx]
                idx += 1

                # print(f"file_path: {file_path}")
                img = PILImage.open(file_path)
                # ⚠️ 注意：上面的一行代码，会有解密插件接管，解密插件会在上传前解密图片 ⚠️
                # 因此，下面的代码是不需要的。 否则，解密再解密会导致图片解密失败。
                # decrypted_img = dencrypt_image_v2(img, get_sha256(password))
                
                # 使用 BytesIO 在内存中保存图像数据
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='PNG')  # 或者 'JPEG'，根据需要选择
                img_byte_arr.seek(0)  # 将指针移回文件开头

                # 准备请求参数和文件
                # payload = {
                #     "key": imgbb_api_key,
                # }
                # # files参数会自动处理multipart/form-data
                # files = {
                #     "image": (os.path.basename(file_path), img_byte_arr, 'image/png'),
                # }

                # # print("正在上传图片...")
                # response = requests.post(url, data=payload, files=files)
                
                # # 检查响应状态码
                # if response.status_code == 200:
                #     result = response.json()
                #     if result['success']:
                #         print(f"图片 {file_path} 上传成功！")
                #         upload_data = result['data']
                #         print(f"upload_data: {upload_data}")
                #         # print(upload_data['url'])
                #         # print(upload_data['thumb']['url'])
                #         output_paths.append(f"{upload_data['url']}|||{upload_data['thumb']['url']}|||{nsfw_prob:.4f}")
                #         # output_thumb_paths.append(upload_data['thumb']['url'])

                #     else:
                #         print(f"图片上传失败: {result['error']['message']}")
                # else:
                #     print(f"请求失败，状态码：{response.status_code}")
                #     print(f"响应内容：{response.text}")


                # 发送请求
                response = requests.post(
                    'https://all4bridge.serv00.net/upload-image-binary',
                    data=img_byte_arr.getvalue(),
                    # headers=headers
                )
                
                # 处理响应
                if response.status_code == 200:
                    result = response.json()
                    print(f"✅ 上传成功!")
                    print(f"原图URL: {result['url']}")
                    print(f"缩略图URL: {result['thumb']}")
                    print(f"文件大小: {result['size']} bytes")
                    output_paths.append(f"{result['url']}|||{result['thumb']}|||{nsfw_prob:.4f}")
                    # return result
                else:
                    print(f"❌ 上传失败: {response.text}")
                    # return None

            
            return (",".join(output_paths),)
        except Exception as e:
            return (f"Upload failed: {str(e)}",)



class UploadAllOutputsToHFDataset:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hf_token": ("STRING", {"default": ""}),
                "dataset_name": ("STRING", {"default": ""}),
                "huggingface_path_in_repo": ("STRING", {"default": ""}),
                "outputs_folder": ("STRING", {"default": folder_paths.get_output_directory()}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("result",)
    FUNCTION = "upload"
    CATEGORY = "utils"

    def upload(self, hf_token, dataset_name, huggingface_path_in_repo, outputs_folder):
        api = HfApi()
        HfFolder.save_token(hf_token)
        if not os.path.exists(outputs_folder):
            return ("Outputs folder does not exist.",)
        files = [os.path.join(outputs_folder, f) for f in os.listdir(outputs_folder) if os.path.isfile(os.path.join(outputs_folder, f))]
        if not files:
            return ("No files to upload.",)
        try:
            for file_path in files:
                path_in_repo = os.path.join(huggingface_path_in_repo, os.path.basename(file_path))

                api.upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=path_in_repo,
                    repo_id=dataset_name,
                    repo_type="dataset",
                    token=hf_token,
                )
            return (f"Uploaded {len(files)} files to {dataset_name}.",)
        except Exception as e:
            return (f"Upload failed: {str(e)}",)

class DownloadFromHFDataset:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hf_token": ("STRING", {"default": ""}),
                "dataset_name": ("STRING", {"default": ""}),
                "download_folder": ("STRING", {"default": folder_paths.get_input_directory()}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("result",)
    FUNCTION = "download"
    CATEGORY = "utils"

    def download(self, hf_token, dataset_name, download_folder):
        api = HfApi()
        HfFolder.save_token(hf_token)
        
        # Create download folder if it doesn't exist
        if not os.path.exists(download_folder):
            os.makedirs(download_folder, exist_ok=True)

        try:
            # List all files in the dataset
            dataset_info = api.list_repo_files(repo_id=dataset_name, repo_type="dataset", token=hf_token)
            
            if not dataset_info:
                return ("No files found in the specified dataset.",)

            downloaded_count = 0
            for file_path in dataset_info:
                # Exclude .gitattributes and other non-data files
                if file_path.startswith("."):
                    continue
                
                # Check if file already exists to avoid re-downloading
                local_file_path = os.path.join(download_folder, os.path.basename(file_path))
                if os.path.exists(local_file_path):
                    print(f"File already exists, skipping: {local_file_path}")
                    continue

                # Download the file
                api.hf_hub_download(
                    repo_id=dataset_name,
                    filename=file_path,
                    repo_type="dataset",
                    local_dir=download_folder,
                    local_dir_use_symlinks=False,
                    token=hf_token
                )
                downloaded_count += 1
                
            return (f"Downloaded {downloaded_count} files to {download_folder}.",)
            
        except Exception as e:
            return (f"Download failed: {str(e)}",)


class UpdateOrder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "outputs": ("STRING", {}),
                "host_update_order": ("STRING",{"default":"https://log.yesky.online/update-submission-urls"}),
                "enable_publish": ("BOOLEAN", {"default": False}),
                "order_id": ("INT", {"default": -1}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "updateorder"
    CATEGORY = "utils"

    def updateorder(self, outputs, host_update_order, enable_publish, order_id):

        print(f"outputs: {outputs}")

        if not outputs:
            return (outputs,)

        parts = outputs.split(',')
        print(f"parts: {parts}")
        urls = [item.split('|||')[0] for item in parts]
        urls_thumb = [item.split('|||')[1] for item in parts]
        nsfw_probs = [item.split('|||')[2] for item in parts]

        print(f"nsfw_probs: {nsfw_probs}")

        # 调用接口，更新userOrders表
        update_data = {
            "id": order_id,
            "published": enable_publish,
            "output_paths": ",".join(urls),
            "output_thumb_paths": ",".join(urls_thumb),
            "nsfw_probs": ",".join([str(prob) for prob in nsfw_probs]),
        }
        print(f"update_data: {update_data}")
        response = requests.post(host_update_order, json=update_data)
        if response.status_code == 200 or response.status_code == 201:
            data = response.json()
            print("请求成功:", data)
        else:
            print(f"请求失败，状态码: {response.status_code}")
            print("错误信息:", response.text)

        return (outputs,)

class SendEmail:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "outputs": ("STRING", {}),
                "ai_host_api": ("STRING", {"default": "https://hengai.pages.dev/view"}),
                "smtp_server": ("STRING", {"default": "mail11.serv00.com"}),
                "smtp_port": ("INT", {"default": 587}),
                "username": ("STRING", {"default": "administrator@all4bridge.serv00.net"}),
                "password": ("STRING", {"default": ""}),
                "from_addr": ("STRING", {"default": "administrator@all4bridge.serv00.net"}),
                "to_addr": ("STRING", {"default": ""}),
                "subject": ("STRING", {"default": "AI Generation Notification"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "send"
    CATEGORY = "utils"

    @staticmethod
    def compress_urls(url_string):
        """
        使用 zlib 压缩字符串，并用 Base64 进行编码，使其成为 URL 安全的字符串。

        Args:
            url_string: 待压缩的原始字符串。

        Returns:
            压缩并编码后的字符串。
        """
        # 将字符串编码为字节
        data_bytes = url_string.encode('utf-8')
        # 使用 zlib 进行压缩
        compressed_data = zlib.compress(data_bytes)
        # 使用 Base64 进行编码，并移除末尾的填充符 '='
        base64_encoded = base64.urlsafe_b64encode(compressed_data).decode('utf-8').rstrip('=')
        return base64_encoded


    def send(self, outputs, ai_host_api, smtp_server, smtp_port, username, password, from_addr, to_addr, subject):
        msg = MIMEMultipart('alternative')
        msg['From'] = from_addr
        msg['To'] = to_addr
        msg['Subject'] = subject

        print(f"outputs: {outputs} from {from_addr} to {to_addr}")

        if not outputs or not to_addr:
            return (outputs,)

        # 定义变量
        # repo_dataset = "Heng365/outputs"  # 仓库路径变量

        # 分割字符串获取文件名列表
        # file_names = [name.strip() for name in outputs.split(",")]

        # # 构建URL列表
        # base_url = f"https://hf-mirror.com/datasets/{repo_dataset}/resolve/main/"
        # urls = [f"{base_url}{file_name}" for file_name in file_names]

        # # 拼接成最终格式
        # str_urls = ",".join(urls)


        # 不用hf datasets 变量了. 直接用outputs
        parts = outputs.split(',')
        urls = [item.split('|||')[0] for item in parts]
        compressed_str_urls = SendEmail.compress_urls(",".join(urls))

        result = f"{ai_host_api}?data={compressed_str_urls}"

        # 纯文本版本
        text = f"""您好！
    
您的AI生成项目已执行完成。
请点击链接查看：{result}
    
如果链接无法点击，请复制到浏览器中打开。



Hello!

Your AI-generated project has been successfully completed.
Please click the link to view it: {result}

If the link is not clickable, please copy it and open it in your browser.

"""
        part1 = MIMEText(text, 'plain')
    
        # HTML版本
        html = f"""<html>
<body>
    <p>您好！</p>
    <p>感谢您关注我们的服务，您的AI生成项目已执行完成。</p>
    <p>请点击链接查看：<a href="{result}">查看<a></p>
    <p><small>如果链接无法点击，请复制以下地址到浏览器中打开：<br>
    {result}</small></p>

<br/>
<br/>
<br/>

<p>Hello!</p>
<p>Thank you for using our service. Your AI-generated project has been successfully completed.</p>
<p>Please click the link below to view it: <a href="{result}">View</a></p>
<p><small>If the link is not clickable, please copy the following address and paste it into your browser:<br>
{result}</small></p>

</body>
</html>"""
        part2 = MIMEText(html, 'html')
        
        # 按照从简单到复杂的顺序添加
        msg.attach(part1)  # 先添加简单版本
        msg.attach(part2)  # 后添加复杂版本


        try:
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(username, password)
            text = msg.as_string()
            server.sendmail(from_addr, to_addr, text)
            server.quit()
            return ("Email sent successfully.",)
        except Exception as e:
            return (f"Failed to send email: {str(e)}",)


NODE_CLASS_MAPPINGS = {
    "UploadAllOutputsToHFDataset": UploadAllOutputsToHFDataset,
    "PushToHFDataset": PushToHFDataset,
    "NSFWFilter": NSFWFilter,
    "PushToImageBB": PushToImageBB,
    "DownloadFromHFDataset": DownloadFromHFDataset,
    "UpdateOrder": UpdateOrder,
    "SendEmail": SendEmail,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "UploadAllOutputsToHFDataset": "Upload outputs to HuggingFace Dataset",
    "PushToHFDataset": "Push Images to HuggingFace Dataset",
    "NSFWFilter": "NSFW Filter",
    "PushToImageBB": "Push Images to ImgBB",
    "UpdateOrder": "Update Order",
    "DownloadFromHFDataset": "Download from HuggingFace Dataset",
    "SendEmail": "Send Email",
}