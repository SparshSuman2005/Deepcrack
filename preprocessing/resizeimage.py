from PIL import Image 
import os

input_folder=r"D:\DeepCrack\dataset\train_img"

output_folder = r"D:\DeepCrack\dataset\train_img_resized"

os.makedirs(output_folder, exist_ok=True)

target_size = (256, 256)

for filename in os.listdir(input_folder):

    if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):

        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        image = Image.open(input_path)

        resized_image = image.resize(target_size, Image.Resampling.BILINEAR)

        resized_image.save(output_path)

        print(f"Resized: {filename}")

print("Done!")