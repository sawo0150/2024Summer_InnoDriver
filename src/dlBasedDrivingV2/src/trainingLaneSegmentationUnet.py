from keras_segmentation.models.unet import mobilenet_unet
from keras_segmentation.data_utils.data_loader import verify_segmentation_dataset

# 데이터셋 검증
verify_segmentation_dataset(
    images_path="/home/innodriver/InnoDriver_ws/Unet_train/Lane segmentation.v5i.png-mask-semantic/train/images/",
    segs_path="/home/innodriver/InnoDriver_ws/Unet_train/Lane segmentation.v5i.png-mask-semantic/train/annotations/",
    n_classes=3
)

verify_segmentation_dataset(
    images_path="/home/innodriver/InnoDriver_ws/Unet_train/Lane segmentation.v5i.png-mask-semantic/valid/images/",
    segs_path="/home/innodriver/InnoDriver_ws/Unet_train/Lane segmentation.v5i.png-mask-semantic/valid/annotations/",
    n_classes=3
)

# MobileNet 기반 U-Net 모델 생성
model = mobilenet_unet(n_classes=3, input_height=224, input_width=224)

# 모델 학습
model.train(
    train_images="/home/innodriver/InnoDriver_ws/Unet_train/Lane segmentation.v5i.png-mask-semantic/train/images/",
    train_annotations="/home/innodriver/InnoDriver_ws/Unet_train/Lane segmentation.v5i.png-mask-semantic/train/annotations/",
    checkpoints_path="/home/innodriver/InnoDriver_ws/Unet_train/checkpoints/mobile_unet",
    epochs=100,
    validate=True,
    val_images="/home/innodriver/InnoDriver_ws/Unet_train/Lane segmentation.v5i.png-mask-semantic/valid/images/",
    val_annotations="/home/innodriver/InnoDriver_ws/Unet_train/Lane segmentation.v5i.png-mask-semantic/valid/annotations/"
)

# # 학습된 모델로 예측
# out = model.predict_segmentation(
#     inp="/home/innodriver/InnoDriver_ws/Unet_train/Lane segmentation.v5i.png-mask-semantic/valid/images/1721101461697221279_jpg.rf.f69d645931f5801ac4d82f3dab2a6d00.jpg",
#     out_fname="/home/innodriver/InnoDriver_ws/Unet_train/predictions/out.png"
# )

# # 결과 시각화
# import matplotlib.pyplot as plt
# plt.imshow(out)
# plt.show()
