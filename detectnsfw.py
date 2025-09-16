import opennsfw2 as n2

# To get the NSFW probability of a single image, provide your image file path,
# or a `PIL.Image.Image` object.
image_handle = "path/to/your/image.jpg"

nsfw_probability = n2.predict_image(image_handle)
print(nsfw_probability)
