import os
import json
import torch
import numpy as np
import cv2
import mediapipe as mp
from torchvision import transforms
from mobilenetV2 import MobileNetV2


class MediaPipeHandProcessor:
    """MediaPipe手部姿态处理器"""

    def __init__(self, static_image_mode=False, max_num_hands=2,
                 min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """
        初始化MediaPipe手部姿态处理器

        Args:
            static_image_mode: 是否静态图像模式
            max_num_hands: 最大检测手数
            min_detection_confidence: 最小检测置信度
            min_tracking_confidence: 最小跟踪置信度
        """
        # 初始化MediaPipe手部解决方案
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # 创建手部检测器
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        # 定义手部关键点连接线
        self.connections = self.mp_hands.HAND_CONNECTIONS

        # 绘制参数
        self.point_color = (0, 0, 255)  # 红色点 (BGR格式)
        self.line_color = (255, 255, 255)  # 红色线 (BGR格式)
        self.point_radius = 5
        self.line_thickness = 2

    def process_frame(self, frame):
        """
        处理单帧图像，检测手部姿态点

        Args:
            frame: OpenCV BGR格式图像

        Returns:
            tuple: (landmarks_image, landmarks_list, success)
                - landmarks_image: 姿态点图像 (黑色背景，红色姿态点)
                - landmarks_list: 检测到的关键点列表
                - success: 是否成功检测到手部
        """
        # 复制原始图像
        image = frame.copy()

        # 转换为RGB格式（MediaPipe需要RGB输入）
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 处理图像，检测手部
        results = self.hands.process(image_rgb)

        # 创建黑色背景图像
        h, w = image.shape[:2]
        landmarks_image = np.zeros((h, w, 3), dtype=np.uint8)

        landmarks_list = []
        success = False

        # 如果检测到手部
        if results.multi_hand_landmarks:
            success = True

            for hand_landmarks in results.multi_hand_landmarks:
                # 提取关键点坐标
                landmarks = []
                for lm in hand_landmarks.landmark:
                    # 将归一化坐标转换为像素坐标
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    landmarks.append((cx, cy))

                landmarks_list.append(landmarks)

                # 在黑色背景上绘制关键点
                for point in landmarks:
                    cx, cy = point
                    cv2.circle(landmarks_image, (cx, cy),
                               self.point_radius, self.point_color, -1)

                # 绘制连接线
                for connection in self.connections:
                    start_idx, end_idx = connection
                    if start_idx < len(landmarks) and end_idx < len(landmarks):
                        start_point = landmarks[start_idx]
                        end_point = landmarks[end_idx]
                        cv2.line(landmarks_image, start_point, end_point,
                                 self.line_color, self.line_thickness)

        return landmarks_image, landmarks_list, success

    def close(self):
        """释放资源"""
        self.hands.close()


class GesturePredictor:
    def __init__(self, model_weights_path="./sixclass/checkpoints/model_best.pth",
                 class_indices_path="./sixclass/class_indices.json", device=None):
        """
        初始化手势识别器

        Args:
            model_weights_path: 模型权重文件路径
            class_indices_path: 类别索引文件路径
            device: 指定设备 (cuda:0 或 cpu)
        """
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # 检查文件是否存在
        assert os.path.exists(model_weights_path), f"权重文件不存在: {model_weights_path}"

        # 加载检查点
        checkpoint = torch.load(model_weights_path, map_location=self.device)

        # 检查检查点中是否包含类别索引
        if 'class_indices' in checkpoint:
            self.class_indict = checkpoint['class_indices']
            print("从检查点加载类别索引")
            print(f"类别索引: {self.class_indict}")
        else:
            # 从外部文件加载类别索引
            assert os.path.exists(class_indices_path), f"类别索引文件不存在: {class_indices_path}"
            with open(class_indices_path, "r") as f:
                self.class_indict = json.load(f)
            print("从外部文件加载类别索引")
            print(f"类别索引: {self.class_indict}")

        # 创建模型
        self.model = MobileNetV2(num_classes=len(self.class_indict)).to(self.device)

        # 从检查点加载模型权重
        if 'model_state_dict' in checkpoint:
            # 检查点包含完整的训练状态
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print("从检查点加载模型状态字典")

            # 打印一些训练信息（如果有的话）
            if 'epoch' in checkpoint:
                print(f"模型训练轮次: {checkpoint['epoch']}")
            if 'best_acc' in checkpoint:
                print(f"模型最佳准确率: {checkpoint['best_acc']:.4f}")
        else:
            # 检查点直接是模型状态字典
            self.model.load_state_dict(checkpoint)
            print("直接加载模型权重")

        self.model.eval()  # 设置为评估模式

    def preprocess_frame(self, frame):
        """
        预处理OpenCV帧

        Args:
            frame: OpenCV BGR格式图像 (H, W, 3)

        Returns:
            torch.Tensor: 预处理后的图像张量，形状为 (1, 3, 224, 224)
        """
        if frame is None:
            raise ValueError("输入帧为空")

        # 确保是BGR格式
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            raise ValueError(f"输入图像形状不正确: {frame.shape}，应为 (H, W, 3)")

        # 将BGR转换为RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 调整大小到256x256
        resized = cv2.resize(rgb_frame, (256, 256))

        # 中心裁剪到224x224
        h, w = resized.shape[:2]
        start_h = (h - 224) // 2
        start_w = (w - 224) // 2
        cropped = resized[start_h:start_h + 224, start_w:start_w + 224]

        # 转换为[0, 1]范围的float32
        normalized = cropped.astype(np.float32) / 255.0

        # 标准化 (使用ImageNet均值/std)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (normalized - mean) / std

        # 转换为PyTorch张量并调整维度顺序: (H, W, C) -> (C, H, W)
        tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).float()

        # 添加batch维度: (C, H, W) -> (1, C, H, W)
        tensor = tensor.unsqueeze(0)

        return tensor

    def predict(self, frame):
        """
        预测单张图像的手势类别和置信度

        Args:
            frame: numpy数组，OpenCV BGR格式图像 (H, W, 3)

        Returns:
            tuple: (predicted_class, confidence, all_probabilities)
                - predicted_class: 预测的类别名称
                - confidence: 预测类别的置信度
                - all_probabilities: 所有类别的概率分布字典
        """
        # 预处理图像
        img_tensor = self.preprocess_frame(frame)

        # 预测
        with torch.no_grad():
            img_tensor = img_tensor.to(self.device)
            output = self.model(img_tensor)

            # 计算softmax概率
            probabilities = torch.nn.functional.softmax(output, dim=1)
            probabilities = probabilities.squeeze().cpu().numpy()  # 转换为numpy数组

            # 获取预测结果
            predicted_idx = np.argmax(probabilities)
            confidence = probabilities[predicted_idx]

            # 修复：处理类别索引可能为整数或字符串的情况
            predicted_class = None
            for key in [str(predicted_idx), predicted_idx]:
                if key in self.class_indict:
                    predicted_class = self.class_indict[key]
                    break

            if predicted_class is None:
                # 如果找不到对应的类别，使用默认值
                predicted_class = f"class_{predicted_idx}"
                print(f"警告: 类别索引 {predicted_idx} 不在类别字典中，使用默认名称: {predicted_class}")
                print(f"可用的类别键: {list(self.class_indict.keys())}")

            # 获取所有类别的概率分布
            all_probs = {}
            for idx in range(len(probabilities)):
                # 尝试不同的键格式
                class_name = None
                for key in [str(idx), idx]:
                    if key in self.class_indict:
                        class_name = self.class_indict[key]
                        break

                if class_name is None:
                    class_name = f"class_{idx}"

                all_probs[class_name] = float(probabilities[idx])

        return predicted_class, confidence, all_probs

    def predict_simple(self, frame):
        """
        简化版预测，只返回类别和置信度

        Args:
            frame: OpenCV BGR格式图像

        Returns:
            tuple: (predicted_class, confidence)
        """
        predicted_class, confidence, _ = self.predict(frame)
        return predicted_class, confidence


def real_time_camera_prediction():
    """
    实时摄像头手势识别
    """
    # 初始化摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("错误: 无法打开摄像头")
        return

    # 设置摄像头分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 初始化MediaPipe手部处理器
    print("初始化MediaPipe手部处理器...")
    hand_processor = MediaPipeHandProcessor(
        static_image_mode=False,
        max_num_hands=1,  # 只检测一只手
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )

    # 初始化手势识别器
    print("初始化手势识别器...")
    predictor = GesturePredictor()
    print(f"使用设备: {predictor.device}")
    print(f"类别数量: {len(predictor.class_indict)}")
    print(f"类别索引: {predictor.class_indict}")

    print("\n手势识别系统已启动!")
    print("按 'q' 键退出")
    print("按 's' 键保存当前帧和姿态点图像")

    # 性能监控
    import time
    frame_count = 0
    fps_start_time = time.time()
    fps = 0

    # 创建保存目录
    save_dir = "./saved_images"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    while True:
        # 读取摄像头帧
        ret, frame = cap.read()
        if not ret:
            print("错误: 无法从摄像头读取帧")
            break

        # 水平翻转图像（镜像效果）
        frame = cv2.flip(frame, 1)

        # 处理图像，获取姿态点图像
        landmarks_image, landmarks_list, success = hand_processor.process_frame(frame)

        # 计算FPS
        frame_count += 1
        if frame_count % 30 == 0:
            fps_end_time = time.time()
            fps = 30 / (fps_end_time - fps_start_time)
            fps_start_time = fps_end_time

        # 如果检测到手部，进行预测
        prediction_result = None
        if success and landmarks_list:
            # 使用姿态点图像进行预测
            try:
                predicted_class, confidence, all_probs = predictor.predict(landmarks_image)
                prediction_result = (predicted_class, confidence)

                # 在原始图像上显示预测结果
                cv2.putText(frame, f"Gesture: {predicted_class}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # # 在姿态点图像上显示预测结果
                # cv2.putText(landmarks_image, f"Gesture: {predicted_class}", (10, 30),
                #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                # cv2.putText(landmarks_image, f"Confidence: {confidence:.2f}", (10, 70),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            except Exception as e:
                print(f"预测错误: {e}")
        else:
            # 未检测到手部
            cv2.putText(frame, "No hand detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # cv2.putText(landmarks_image, "No hand detected", (10, 30),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 在原始图像上显示FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        #
        # # 在姿态点图像上显示FPS
        # cv2.putText(landmarks_image, f"FPS: {fps:.1f}", (10, 110),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # 显示图像
        cv2.imshow('Camera - Original Image', frame)
        cv2.imshow('Camera - Landmarks Image', landmarks_image)

        # 等待按键
        key = cv2.waitKey(1) & 0xFF

        # 按 'q' 键退出
        if key == ord('q'):
            break

        # 按 's' 键保存图像
        elif key == ord('s'):
            timestamp = time.strftime("%Y%m%d_%H%M%S")

            # 保存原始图像
            original_path = os.path.join(save_dir, f"original_{timestamp}.jpg")
            cv2.imwrite(original_path, frame)

            # 保存姿态点图像
            landmarks_path = os.path.join(save_dir, f"landmarks_{timestamp}.jpg")
            cv2.imwrite(landmarks_path, landmarks_image)

            print(f"图像已保存: {original_path}, {landmarks_path}")

            # 如果检测到手部，保存预测结果
            if prediction_result:
                predicted_class, confidence = prediction_result
                result_path = os.path.join(save_dir, f"result_{timestamp}.txt")
                with open(result_path, 'w') as f:
                    f.write(f"Timestamp: {timestamp}\n")
                    f.write(f"Predicted Gesture: {predicted_class}\n")
                    f.write(f"Confidence: {confidence:.4f}\n")
                print(f"预测结果已保存: {result_path}")

    # 释放资源
    cap.release()
    hand_processor.close()
    cv2.destroyAllWindows()
    print("\n手势识别系统已关闭")


def predict_image(image_path="./test.jpg"):
    """
    使用指定图像文件测试手势识别器

    Args:
        image_path: 测试图像文件路径
    """
    # 检查图像文件是否存在
    if not os.path.exists(image_path):
        print(f"错误: 图像文件不存在: {image_path}")
        print("请确保当前目录下存在 test.jpg 文件")
        return

    # 初始化MediaPipe手部处理器
    print("初始化MediaPipe手部处理器...")
    hand_processor = MediaPipeHandProcessor(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5
    )

    # 初始化识别器
    print("初始化手势识别器...")
    predictor = GesturePredictor()
    print(f"使用设备: {predictor.device}")
    print(f"类别数量: {len(predictor.class_indict)}")
    print(f"类别索引: {predictor.class_indict}")

    # 加载测试图像
    print(f"\n加载测试图像: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"错误: 无法加载图像: {image_path}")
        return

    print(f"图像尺寸: {image.shape}")

    # 处理图像，获取姿态点图像（添加计时）
    import time
    print("\n开始处理手势姿态点图...")
    start_landmark_time = time.time()
    landmarks_image, landmarks_list, success = hand_processor.process_frame(image)
    end_landmark_time = time.time()
    landmark_process_time = (end_landmark_time - start_landmark_time) * 1000  # 转换为毫秒

    print(f"手势姿态点图处理时间: {landmark_process_time:.2f} ms")

    if not success:
        print("警告: 未检测到手部姿态点")
        # 显示原始图像
        cv2.imshow("测试图像", image)
        print("按任意键关闭图像窗口...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # 显示原始图像和姿态点图像
    combined = np.hstack([image, landmarks_image])
    cv2.imshow("原始图像 (左) vs 姿态点图像 (右)", combined)
    print("按任意键关闭图像窗口并继续...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 测试推理时间
    print("\n开始MobileNet推理...")
    start_inference_time = time.time()

    # 单次推理
    predicted_class, confidence, all_probs = predictor.predict(landmarks_image)

    end_inference_time = time.time()
    inference_time = (end_inference_time - start_inference_time) * 1000  # 转换为毫秒

    print(f"MobileNet推理时间: {inference_time:.2f} ms")
    print(f"总处理时间: {landmark_process_time + inference_time:.2f} ms")

    # 多次推理测试平均时间
    print("\n测试多次推理性能...")
    test_iterations = 10
    total_inference_time = 0

    for i in range(test_iterations):
        start_iter_time = time.time()
        result = predictor.predict(landmarks_image)
        end_iter_time = time.time()
        iter_time = (end_iter_time - start_iter_time) * 1000
        total_inference_time += iter_time

        if i == 0:  # 只在第一次显示结果
            predicted_class, confidence, all_probs = result

    avg_inference_time = total_inference_time / test_iterations
    print(f"平均推理时间: {avg_inference_time:.2f} ms")
    print(f"推理FPS: {1000 / avg_inference_time:.2f}")

    # 显示预测结果
    print("\n" + "=" * 50)
    print("预测结果汇总:")
    print("=" * 50)
    print(f"手势姿态点图处理时间: {landmark_process_time:.2f} ms")
    print(f"MobileNet推理时间: {inference_time:.2f} ms")
    print(f"单次总处理时间: {landmark_process_time + inference_time:.2f} ms")
    print(f"平均推理时间 ({test_iterations}次): {avg_inference_time:.2f} ms")
    print(f"预测类别: {predicted_class}")
    print(f"置信度: {confidence:.4f}")

    print("\n所有类别概率:")
    for class_name, prob in all_probs.items():
        print(f"  {class_name}: {prob:.4f}")

    # 可视化结果
    result_image = landmarks_image.copy()
    cv2.putText(result_image, f"Prediction: {predicted_class}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(result_image, f"Confidence: {confidence:.4f}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(result_image, f"Landmark Time: {landmark_process_time:.1f}ms", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(result_image, f"Inference Time: {inference_time:.1f}ms", (10, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("预测结果", result_image)
    print("\n按任意键退出...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 关闭MediaPipe处理器
    hand_processor.close()

# 使用示例
if __name__ == "__main__":
    print("=" * 50)
    print("手势识别系统")
    print("=" * 50)
    print("1. 实时摄像头手势识别")
    print("2. 测试单张图像")
    print("3. 退出")

    choice = input("\n请选择模式 (1/2/3): ").strip()

    if choice == "1":
        real_time_camera_prediction()
    elif choice == "2":
        test_image_path = "./test.png"
        if not os.path.exists(test_image_path):
            test_image_path = input("请输入测试图像路径: ").strip()
        predict_image(test_image_path)
    elif choice == "3":
        print("程序退出")
    else:
        print("无效选择，程序退出")