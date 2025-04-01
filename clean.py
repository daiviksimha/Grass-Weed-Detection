import os

def clean_dataset(image_dir, label_dir):
    # List all image files in the image directory
    image_filenames = os.listdir(image_dir)

    for filename in image_filenames:
        # Construct the full path for the image and its corresponding label file
        img_path = os.path.join(image_dir, filename)
        label_path = os.path.join(label_dir, filename.replace('.jpg', '.txt'))  # Adjust if your images have a different extension

        # Check if the label file exists
        if not os.path.exists(label_path):
            print(f"Label file does not exist for {filename}. Deleting image.")
            os.remove(img_path)  # Delete the image
            continue

        # Check if the label file is empty
        if os.path.getsize(label_path) == 0:
            print(f"Label file is empty for {filename}. Deleting image.")
            os.remove(img_path)  # Delete the image

if __name__ == "__main__":
    # Set your directories here
    image_directory = r'D:\Projects\mini project\project\data\train\images'
    label_directory = r'D:\Projects\mini project\project\data\train\labels'

    clean_dataset(image_directory, label_directory)