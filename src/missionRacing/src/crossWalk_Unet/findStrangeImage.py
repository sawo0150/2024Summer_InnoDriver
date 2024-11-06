import os
from PIL import Image

def check_and_resize_images(images_path, annotations_path, target_size=(224, 224)):
    image_files = os.listdir(images_path)
    annotation_files = os.listdir(annotations_path)
    resized_files = []

    for image_file in image_files:
        image_path = os.path.join(images_path, image_file)
        annotation_file = image_file.replace('.jpg', '.png')
        annotation_path = os.path.join(annotations_path, annotation_file)
        
        if os.path.exists(annotation_path):
            image = Image.open(image_path)
            annotation = Image.open(annotation_path)
            
            if image.size != annotation.size:
                print(f"Resizing {image_path} and {annotation_path} to {target_size}")
                image = image.resize(target_size, Image.Resampling.LANCZOS)
                annotation = annotation.resize(target_size, Image.Resampling.NEAREST)
                
                image.save(image_path)
                annotation.save(annotation_path)
                resized_files.append((image_path, annotation_path))

    return resized_files

def find_and_resize_mismatched_files(dataset_paths):
    for dataset, paths in dataset_paths.items():
        print(f"Checking {dataset} dataset")
        images_path = paths['images']
        annotations_path = paths['annotations']

        resized_files = check_and_resize_images(images_path, annotations_path)
        
        if resized_files:
            print(f"Resized files in {dataset} dataset:")
            for image_path, annotation_path in resized_files:
                print(f"{image_path}, {annotation_path}")
        else:
            print(f"No mismatched files found in {dataset} dataset.")

# 경로 설정
dataset_paths = {
    "train": {
        "images": "/home/innodriver/InnoDriver_ws/Unet_crosswalk_train/Unet_crosswalk_Detection.v1i.png-mask-semantic/train/images/",
        "annotations": "/home/innodriver/InnoDriver_ws/Unet_crosswalk_train/Unet_crosswalk_Detection.v1i.png-mask-semantic/train/annotations/"
    },
    "valid": {
        "images": "/home/innodriver/InnoDriver_ws/Unet_crosswalk_train/Unet_crosswalk_Detection.v1i.png-mask-semantic/valid/images/",
        "annotations": "/home/innodriver/InnoDriver_ws/Unet_crosswalk_train/Unet_crosswalk_Detection.v1i.png-mask-semantic/valid/annotations/"
    },
    "test": {
        "images": "/home/innodriver/InnoDriver_ws/Unet_crosswalk_train/Unet_crosswalk_Detection.v1i.png-mask-semantic/test/images/",
        "annotations": "/home/innodriver/InnoDriver_ws/Unet_crosswalk_train/Unet_crosswalk_Detection.v1i.png-mask-semantic/test/annotations/"
    }
}

# 이미지와 주석 파일 크기를 확인하고 224x224로 조정
find_and_resize_mismatched_files(dataset_paths)
