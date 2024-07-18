import os
from PIL import Image

def check_image_sizes(images_path, annotations_path):
    image_files = os.listdir(images_path)
    annotation_files = os.listdir(annotations_path)
    mismatched_files = []

    for image_file in image_files:
        image_path = os.path.join(images_path, image_file)
        annotation_file = image_file.replace('.jpg', '.png')
        annotation_path = os.path.join(annotations_path, annotation_file)
        
        if os.path.exists(annotation_path):
            image = Image.open(image_path)
            annotation = Image.open(annotation_path)
            
            if image.size != annotation.size:
                mismatched_files.append((image_path, annotation_path))

    return mismatched_files

def find_and_delete_mismatched_files(dataset_paths):
    for dataset, paths in dataset_paths.items():
        print(f"Checking {dataset} dataset")
        images_path = paths['images']
        annotations_path = paths['annotations']

        mismatched_files = check_image_sizes(images_path, annotations_path)
        
        if mismatched_files:
            print(f"Mismatched files found in {dataset} dataset:")
            for image_path, annotation_path in mismatched_files:
                print(f"{image_path}, {annotation_path}")

            delete_files = input("Do you want to delete these files? (y/n): ")
            if delete_files.lower() == 'y':
                for image_path, annotation_path in mismatched_files:
                    os.remove(image_path)
                    os.remove(annotation_path)
                    print(f"Deleted: {image_path}, {annotation_path}")
            else:
                print("Files were not deleted.")
        else:
            print(f"No mismatched files found in {dataset} dataset.")

# 경로 설정
dataset_paths = {
    "train": {
        "images": "/home/innodriver/InnoDriver_ws/Unet_train/Lane segmentation.v5i.png-mask-semantic/train/images/",
        "annotations": "/home/innodriver/InnoDriver_ws/Unet_train/Lane segmentation.v5i.png-mask-semantic/train/annotations/"
    },
    "valid": {
        "images": "/home/innodriver/InnoDriver_ws/Unet_train/Lane segmentation.v5i.png-mask-semantic/valid/images/",
        "annotations": "/home/innodriver/InnoDriver_ws/Unet_train/Lane segmentation.v5i.png-mask-semantic/valid/annotations/"
    },
    "test": {
        "images": "/home/innodriver/InnoDriver_ws/Unet_train/Lane segmentation.v5i.png-mask-semantic/test/images/",
        "annotations": "/home/innodriver/InnoDriver_ws/Unet_train/Lane segmentation.v5i.png-mask-semantic/test/annotations/"
    }
}

# 이상한 데이터셋 찾고 삭제
find_and_delete_mismatched_files(dataset_paths)
