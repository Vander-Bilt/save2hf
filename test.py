import zlib
import base64

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

# 示例字符串
original_string = "https://i.ibb.co/fWYMQd5/Comfy-UI-00001.png"

compressed_string = compress_urls(original_string)
print(f"原始字符串长度: {len(original_string)}")
print(f"压缩后字符串长度: {len(compressed_string)}")
print(f"压缩后的 URL 参数: {compressed_string}")
