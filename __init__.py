import os
from huggingface_hub import HfApi, HfFolder
import folder_paths
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication


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


class SendEmail:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "outputs": ("STRING", {}),
                "smtp_server": ("STRING", {"default": "smtp.example.com"}),
                "smtp_port": ("INT", {"default": 587}),
                "username": ("STRING", {"default": ""}),
                "password": ("STRING", {"default": ""}),
                "from_addr": ("STRING", {"default": ""}),
                "to_addr": ("STRING", {"default": ""}),
                "subject": ("STRING", {"default": "ComfyUI Notification"}),
                "body": ("STRING", {"default": "Attached are the files."}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "send"
    CATEGORY = "utils"

    def send(self, outputs, smtp_server, smtp_port, username, password, from_addr, to_addr, subject):
        msg = MIMEMultipart()
        msg['From'] = from_addr
        msg['To'] = to_addr
        msg['Subject'] = subject

        # msg.attach(MIMEText(body, 'plain'))


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
    "DownloadFromHFDataset": DownloadFromHFDataset,
    "SendEmail": SendEmail,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "UploadAllOutputsToHFDataset": "Upload outputs to HuggingFace Dataset",
    "PushToHFDataset": "Push Images to HuggingFace Dataset",
    "DownloadFromHFDataset": "Download from HuggingFace Dataset",
    "SendEmail": "Send Email",
}