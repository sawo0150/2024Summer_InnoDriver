import os
import shutil

def restructure_data(base_path):
    for split in ['train', 'valid', 'test']:
        image_dir = os.path.join(base_path, split, 'images')
        annotation_dir = os.path.join(base_path, split, 'annotations')
        
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(annotation_dir, exist_ok=True)
        
        for file in os.listdir(os.path.join(base_path, split)):
            if file.endswith('.jpg'):
                shutil.move(os.path.join(base_path, split, file), os.path.join(image_dir, file))
            elif file.endswith('_mask.png'):
                new_file_name = file.replace('_mask.png', '.png')
                shutil.move(os.path.join(base_path, split, file), os.path.join(annotation_dir, new_file_name))

base_path = '/home/innodriver/InnoDriver_ws/Unet_train/Lane segmentation.v5i.png-mask-semantic'
restructure_data(base_path)
