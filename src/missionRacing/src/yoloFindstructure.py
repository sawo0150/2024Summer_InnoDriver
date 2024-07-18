import os
import tensorflow as tf

# 경로 설정
tf_model_path = '/home/innodriver/InnoDriver_ws/yolo_train/best_saved_model'

# 모델 로드
model = tf.saved_model.load(tf_model_path)
infer = model.signatures['serving_default']

# 입력 및 출력 정보 확인
print("Input keys:", infer.structured_input_signature)
print("Output keys:", infer.structured_outputs)

# 더미 입력 데이터 생성
input_tensor = tf.random.uniform([1, 640, 640, 3], dtype=tf.float32)

# 추론 실행
detections = infer(input_tensor)

# 출력 구조 확인
for key, value in detections.items():
    print(f"Output Key: {key}")
    print(f"Shape: {value.shape}")
    print(f"Data: {value.numpy()}")

# 클래스 개수 확인
output_tensor = detections['output_0']
num_classes = output_tensor.shape[-1] - 5  # 5는 (x, y, width, height, confidence)
print(f"Number of classes: {num_classes}")
