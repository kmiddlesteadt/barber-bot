import os
from PIL import Image

def check_and_remove_corrupted(directory):
    corrupted = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                filepath = os.path.join(root, file)
                try:
                    with Image.open(filepath) as img:
                        img.convert("RGB")
                except Exception as e:
                    corrupted.append(filepath)
                    print(f"Removing corrupted image: {filepath} - {e}")
                    os.remove(filepath)
    return corrupted

if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "training", "face_shapes")
    corrupted_images = check_and_remove_corrupted(data_dir)
    print(f"Removed {len(corrupted_images)} corrupted images.")