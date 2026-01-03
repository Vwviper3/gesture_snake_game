import random
import sys
import time
import pygame
from pygame.locals import *
from collections import deque
import threading
import queue
from typing import Optional, Tuple, List, Deque
import cv2
import numpy as np
import mediapipe as mp
import torch
import json
import os
from mobilenetV2 import MobileNetV2

# 基础设置
SCREEN_HEIGHT = 480
SCREEN_WIDTH = 600
SIZE = 20  # 小方格大小
LINE_WIDTH = 1

# 游戏区域的坐标范围
AREA_X = (0, SCREEN_WIDTH // SIZE - 1)  # 0是左边界，1是右边界
AREA_Y = (2, SCREEN_HEIGHT // SIZE - 1)

# 食物的分值+颜色
FOOD_STYLE_LIST = [
    (10, (255, 100, 100)),  # 红色食物，10分
    (20, (100, 255, 100)),  # 绿色食物，20分
    (30, (100, 100, 255))  # 蓝色食物，30分
]

# 颜色定义
LIGHT = (100, 100, 100)
DARK = (200, 200, 200)
BLACK = (0, 0, 0)
RED = (200, 30, 30)
BACKGROUND = (40, 40, 60)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)  # 手势控制状态颜色
YELLOW = (255, 255, 0)  # 警告颜色
BLUE = (0, 120, 255)  # 状态信息颜色
CYAN = (0, 255, 255)  # 手势识别状态颜色
MAGENTA = (255, 0, 255)  # 高亮颜色
ORANGE = (255, 165, 0)  # 连续手势检测状态

# 控制模式
CONTROL_KEYBOARD = 0
CONTROL_GESTURE = 1


# 性能监控
class PerformanceMonitor:
    """性能监控器，用于测量FPS和延迟"""

    def __init__(self):
        self.fps_history = []
        self.latency_history = []
        self.max_history_size = 100
        self.last_frame_time = time.time()
        self.frame_count = 0
        self.current_fps = 0
        self.total_latency = 0
        self.latency_count = 0

    def update_frame(self):
        """更新帧率计算"""
        current_time = time.time()
        elapsed = current_time - self.last_frame_time
        self.frame_count += 1

        if elapsed >= 1.0:  # 每秒计算一次FPS
            self.current_fps = self.frame_count
            self.fps_history.append(self.current_fps)
            if len(self.fps_history) > self.max_history_size:
                self.fps_history.pop(0)
            self.frame_count = 0
            self.last_frame_time = current_time

    def record_latency(self, latency_ms: float):
        """记录延迟"""
        self.latency_history.append(latency_ms)
        if len(self.latency_history) > self.max_history_size:
            self.latency_history.pop(0)
        self.total_latency += latency_ms
        self.latency_count += 1

    def get_avg_fps(self) -> float:
        """获取平均FPS"""
        if not self.fps_history:
            return 0
        return sum(self.fps_history) / len(self.fps_history)

    def get_avg_latency(self) -> float:
        """获取平均延迟"""
        if not self.latency_history:
            return 0
        return sum(self.latency_history) / len(self.latency_history)

    def get_current_fps(self) -> float:
        """获取当前FPS"""
        return self.current_fps


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
                ▪ predicted_class: 预测的类别名称

                ▪ confidence: 预测类别的置信度

                ▪ all_probabilities: 所有类别的概率分布字典

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


class CameraDisplay:
    """摄像头显示窗口"""

    def __init__(self, window_name="Camera Feed", width=640, height=480):
        self.window_name = window_name
        self.width = width
        self.height = height
        self.last_frame = None
        self.last_landmarks = None
        self.last_prediction = None
        self.last_confidence = 0.0
        self.last_hand_detected = False
        self.frame_queue = queue.Queue(maxsize=1)  # 只保留最新的一帧
        self.running = False
        self.display_thread = None

    def start(self):
        """启动显示线程"""
        if self.running:
            return

        self.running = True
        self.display_thread = threading.Thread(target=self._display_loop, daemon=True)
        self.display_thread.start()
        print(f"摄像头显示窗口已启动: {self.window_name}")

    def stop(self):
        """停止显示线程"""
        self.running = False
        if self.display_thread:
            self.display_thread.join(timeout=2.0)
        cv2.destroyWindow(self.window_name)
        print(f"摄像头显示窗口已关闭: {self.window_name}")

    def update_frame(self, frame, landmarks_image, prediction_info):
        """更新显示帧"""
        if not self.running:
            return

        try:
            # 将信息打包
            display_info = (frame, landmarks_image, prediction_info)
            # 清空队列，只保留最新帧
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    break
            self.frame_queue.put_nowait(display_info)
        except queue.Full:
            # 队列已满，丢弃最旧的一帧
            try:
                self.frame_queue.get_nowait()
                self.frame_queue.put_nowait((frame, landmarks_image, prediction_info))
            except:
                pass

    def _display_loop(self):
        """显示循环"""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.width, self.height)

        while self.running:
            try:
                # 从队列获取显示信息
                frame, landmarks_image, prediction_info = self.frame_queue.get(timeout=0.5)
                self.last_frame = frame
                self.last_landmarks = landmarks_image

                if prediction_info:
                    self.last_prediction, self.last_confidence, self.last_hand_detected = prediction_info

                # 创建显示图像
                display_image = self._create_display_image(frame, landmarks_image, prediction_info)

                # 显示图像
                cv2.imshow(self.window_name, display_image)

                # 处理OpenCV事件
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC键退出
                    self.running = False
                    break

            except queue.Empty:
                # 队列为空，使用上一次的图像
                if self.last_frame is not None and self.last_landmarks is not None:
                    display_image = self._create_display_image(
                        self.last_frame,
                        self.last_landmarks,
                        (self.last_prediction, self.last_confidence, self.last_hand_detected)
                    )
                    cv2.imshow(self.window_name, display_image)
                    cv2.waitKey(1)
            except Exception as e:
                print(f"摄像头显示错误: {e}")
                time.sleep(0.1)

    def _create_display_image(self, frame, landmarks_image, prediction_info):
        """创建显示图像"""
        if frame is None or landmarks_image is None:
            # 创建黑色背景
            display_image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            cv2.putText(display_image, "No camera feed", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return display_image

        # 调整图像大小
        frame_resized = cv2.resize(frame, (self.width // 2, self.height))
        landmarks_resized = cv2.resize(landmarks_image, (self.width // 2, self.height))

        # 水平拼接图像
        display_image = np.hstack([frame_resized, landmarks_resized])

        # 添加分隔线
        cv2.line(display_image, (self.width // 2, 0), (self.width // 2, self.height), (0, 255, 0), 2)

        # 添加标签
        cv2.putText(display_image, "Original", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display_image, "Landmarks", (self.width // 2 + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # 如果有预测信息，添加到图像上
        if prediction_info:
            pred_class, confidence, hand_detected = prediction_info

            # 在手部检测状态
            if hand_detected:
                hand_text = "Hand Detected"
                hand_color = (0, 255, 0)
            else:
                hand_text = "No Hand"
                hand_color = (0, 0, 255)

            cv2.putText(display_image, hand_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, hand_color, 2)

            # 在手势识别结果
            if pred_class:
                pred_text = f"Gesture: {pred_class}"
                conf_text = f"Confidence: {confidence:.2f}"
                cv2.putText(display_image, pred_text, (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(display_image, conf_text, (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                # 在landmarks图像上也显示结果
                cv2.putText(display_image, pred_text, (self.width // 2 + 10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(display_image, conf_text, (self.width // 2 + 10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        return display_image


class GestureRecognizer:
    """手势识别器（多线程版本）- 使用MediaPipe和深度学习模型"""

    def __init__(self, gesture_queue: queue.Queue, performance_monitor: PerformanceMonitor,
                 camera_display: CameraDisplay = None,
                 model_weights_path="./sixclass/checkpoints/model_best.pth",
                 class_indices_path="./sixclass/class_indices.json"):
        self.gesture_queue = gesture_queue
        self.performance_monitor = performance_monitor
        self.camera_display = camera_display
        self.running = False
        self.thread = None
        self.cap = None

        # 模型路径
        self.model_weights_path = model_weights_path
        self.class_indices_path = class_indices_path

        # 调试信息
        self.last_gesture = "无手势"
        self.last_confidence = 0.0
        self.last_hand_detected = False
        self.processing_time = 0.0
        self.last_frame = None
        self.last_landmarks_image = None

        # 手势到方向的映射
        self.gesture_to_direction = {
            "up": (0, -1),
            "down": (0, 1),
            "left": (-1, 0),
            "right": (1, 0),
            "forward": (0, 0),  # 开始/继续信号
            "wave": (0, 0)  # 暂停信号
        }

        # 状态变量
        self.hand_processor = None
        self.predictor = None
        self.model_loaded = False

    def _initialize_components(self):
        """初始化MediaPipe和手势识别器组件"""
        try:
            # 初始化MediaPipe手部处理器
            self.hand_processor = self._create_hand_processor()

            # 初始化手势识别器
            self.predictor = GesturePredictor(
                model_weights_path=self.model_weights_path,
                class_indices_path=self.class_indices_path
            )
            self.model_loaded = True
            print("手势识别组件初始化成功")
            return True
        except Exception as e:
            print(f"手势识别组件初始化失败: {e}")
            import traceback
            traceback.print_exc()
            self.model_loaded = False
            return False

    def _create_hand_processor(self):
        """创建MediaPipe手部处理器实例"""
        # 初始化MediaPipe手部解决方案
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles

        # 创建手部检测器
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,  # 只检测一只手
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        class SimpleHandProcessor:
            def __init__(self, hands_obj, connections, point_color=(0, 0, 255), line_color=(255, 255, 255)):
                self.hands = hands_obj
                self.connections = connections
                self.point_color = point_color
                self.line_color = line_color
                self.point_radius = 5
                self.line_thickness = 2

            def process_frame(self, frame):
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

        return SimpleHandProcessor(hands, mp_hands.HAND_CONNECTIONS)

    def _cleanup_components(self):
        """清理组件资源"""
        if self.hand_processor:
            try:
                self.hand_processor.close()
            except:
                pass
            self.hand_processor = None

        if self.cap and hasattr(self.cap, 'release'):
            try:
                self.cap.release()
            except:
                pass
            self.cap = None

        self.predictor = None
        self.model_loaded = False
        print("手势识别组件已清理")

    def start(self):
        """启动手势识别线程"""
        if self.running:
            print("手势识别线程已在运行中")
            return False

        # 清理旧组件
        self._cleanup_components()

        # 初始化新组件
        if not self._initialize_components():
            print("警告: 组件初始化失败，使用键盘模拟手势")
            self._start_simulation()
            return False

        self.running = True
        self.thread = threading.Thread(target=self._gesture_recognition_loop, daemon=True)
        self.thread.start()
        print("手势识别线程已启动")
        return True

    def stop(self):
        """停止手势识别线程"""
        if not self.running:
            print("手势识别线程未运行")
            return

        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
            self.thread = None

        self._cleanup_components()
        print("手势识别线程已停止")

    def _gesture_recognition_loop(self):
        """手势识别主循环（在独立线程中运行）"""
        try:
            # 初始化摄像头
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("警告: 无法打开摄像头，切换到模拟模式")
                self._simulate_gesture_loop()
                return

            print("摄像头已打开，开始手势识别")

            # 设置摄像头分辨率
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            while self.running:
                # 记录开始时间
                frame_start_time = time.time()

                # 读取摄像头帧
                ret, frame = self.cap.read()
                if not ret:
                    print("无法从摄像头读取帧")
                    time.sleep(0.01)
                    continue

                # 水平翻转图像（镜像效果）
                frame = cv2.flip(frame, 1)

                # 处理图像，获取姿态点图像
                landmarks_image, landmarks_list, success = self.hand_processor.process_frame(frame)

                # 更新手部检测状态
                self.last_hand_detected = success
                self.last_frame = frame
                self.last_landmarks_image = landmarks_image

                direction = None
                confidence = 0.0
                gesture_type = None

                if success and landmarks_list and self.model_loaded:
                    try:
                        # 使用姿态点图像进行预测
                        start_predict_time = time.time()
                        predicted_class, conf = self.predictor.predict_simple(landmarks_image)
                        self.processing_time = (time.time() - start_predict_time) * 1000

                        # 更新调试信息
                        self.last_gesture = predicted_class
                        self.last_confidence = conf

                        # 记录端到端延迟
                        end_to_end_latency = (time.time() - frame_start_time) * 1000
                        self.performance_monitor.record_latency(end_to_end_latency)

                        # 将手势映射为方向
                        if predicted_class in self.gesture_to_direction:
                            direction = self.gesture_to_direction[predicted_class]
                            confidence = conf
                            gesture_type = predicted_class

                            # 将结果放入队列
                            self.gesture_queue.put((direction, confidence, time.time(), gesture_type))
                            print(
                                f"检测到手势: {gesture_type}, 置信度: {confidence:.2f}, 处理时间: {self.processing_time:.1f}ms")

                    except Exception as e:
                        print(f"预测错误: {e}")

                # 更新摄像头显示
                if self.camera_display:
                    prediction_info = (gesture_type, confidence, success) if success else (None, 0.0, False)
                    self.camera_display.update_frame(frame, landmarks_image, prediction_info)

                # 更新FPS
                self.performance_monitor.update_frame()

                # 降低CPU使用率
                time.sleep(0.01)

        except Exception as e:
            print(f"手势识别线程出错: {e}")
            import traceback
            traceback.print_exc()
            # 如果摄像头出错，切换到模拟模式
            self._start_simulation()

    def _start_simulation(self):
        """启动模拟手势识别"""
        self.running = True
        self.thread = threading.Thread(target=self._simulate_gesture_loop, daemon=True)
        self.thread.start()
        print("启动手势模拟模式")

    def _simulate_gesture_loop(self):
        """模拟手势识别（用于调试或摄像头不可用时）"""
        print("使用键盘模拟手势识别")
        print("模拟控制键: I(上), K(下), J(左), L(右), U(开始), O(暂停)")

        while self.running:
            frame_start_time = time.time()

            # 检测模拟按键
            keys = pygame.key.get_pressed()
            direction = None
            gesture_type = None

            if keys[K_i]:  # 上
                direction = (0, -1)
                gesture_type = "up"
            elif keys[K_k]:  # 下
                direction = (0, 1)
                gesture_type = "down"
            elif keys[K_j]:  # 左
                direction = (-1, 0)
                gesture_type = "left"
            elif keys[K_l]:  # 右
                direction = (1, 0)
                gesture_type = "right"
            elif keys[K_u]:  # 开始/继续
                direction = (0, 0)
                gesture_type = "forward"
            elif keys[K_o]:  # 暂停
                direction = (0, 0)
                gesture_type = "wave"

            if direction is not None:
                # 记录延迟
                latency = (time.time() - frame_start_time) * 1000
                self.performance_monitor.record_latency(latency)

                # 更新调试信息
                self.last_gesture = gesture_type
                self.last_confidence = 0.9
                self.last_hand_detected = True
                self.processing_time = 1.0

                # 创建模拟图像
                if self.camera_display:
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(frame, f"Simulated: {gesture_type}", (50, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    landmarks_image = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(landmarks_image, f"Gesture: {gesture_type}", (50, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    self.camera_display.update_frame(frame, landmarks_image, (gesture_type, 0.9, True))

                # 放入队列
                self.gesture_queue.put((direction, 0.9, time.time(), gesture_type))
                print(f"模拟手势: {gesture_type}")

            # 更新FPS
            self.performance_monitor.update_frame()
            time.sleep(0.1)  # 降低模拟速度，避免过快


def print_text(screen, font, x, y, text, color=WHITE):
    """在屏幕上绘制文本"""
    text_surface = font.render(text, True, color)
    screen.blit(text_surface, (x, y))


def init_snake():
    """初始化蛇的位置"""
    snake = deque()
    # 蛇初始为3个格子，水平排列
    snake.append((2, AREA_Y[0]))
    snake.append((1, AREA_Y[0]))
    snake.append((0, AREA_Y[0]))
    return snake


def create_food(snake):
    """生成食物位置，避开蛇身"""
    while True:
        food_x = random.randint(AREA_X[0], AREA_X[1])
        food_y = random.randint(AREA_Y[0], AREA_Y[1])

        # 如果食物不在蛇身上，返回位置
        if (food_x, food_y) not in snake:
            return food_x, food_y


def get_random_food_style():
    """随机获取食物样式（分值+颜色）"""
    return random.choice(FOOD_STYLE_LIST)


def draw_grid(screen):
    """绘制游戏网格"""
    # 绘制竖线
    for x in range(SIZE, SCREEN_WIDTH, SIZE):
        pygame.draw.line(
            screen, BLACK,
            (x, AREA_Y[0] * SIZE),
            (x, SCREEN_HEIGHT),
            LINE_WIDTH
        )

    # 绘制横线
    for y in range(AREA_Y[0] * SIZE, SCREEN_HEIGHT, SIZE):
        pygame.draw.line(
            screen, BLACK,
            (0, y),
            (SCREEN_WIDTH, y),
            LINE_WIDTH
        )


def draw_snake(screen, snake):
    """绘制蛇"""
    for segment in snake:
        pygame.draw.rect(
            screen, DARK,
            (segment[0] * SIZE + LINE_WIDTH,
             segment[1] * SIZE + LINE_WIDTH,
             SIZE - LINE_WIDTH * 2,
             SIZE - LINE_WIDTH * 2),
            0
        )


def draw_food(screen, food_pos, food_color):
    """绘制食物"""
    pygame.draw.rect(
        screen, food_color,
        (food_pos[0] * SIZE, food_pos[1] * SIZE, SIZE, SIZE),
        0
    )


def reset_game():
    """重置游戏状态"""
    snake = init_snake()
    food = create_food(snake)
    food_style = get_random_food_style()
    direction = (1, 0)  # 初始向右移动
    score = 0
    return snake, food, food_style, direction, score


class GestureHistory:
    """手势历史记录器，用于连续手势检测"""

    def __init__(self, window_size=1.0, required_consecutive=3):
        """
        初始化手势历史记录器

        Args:
            window_size: 时间窗口大小（秒）
            required_consecutive: 需要的连续手势数量
        """
        self.window_size = window_size
        self.required_consecutive = required_consecutive
        self.history: Deque[Tuple[str, float]] = deque()  # (gesture_type, timestamp)
        self.last_triggered_gesture = None
        self.last_triggered_time = 0

    def add_gesture(self, gesture_type: str, timestamp: float):
        """添加新手势到历史记录"""
        if gesture_type:  # 只记录有效手势
            self.history.append((gesture_type, timestamp))

        # 清理超过时间窗口的旧记录
        while self.history and timestamp - self.history[0][1] > self.window_size:
            self.history.popleft()

    def check_consecutive_gesture(self, current_time: float, min_interval: float = 0.3) -> Optional[str]:
        """
        检查是否有连续相同的手势

        Args:
            current_time: 当前时间
            min_interval: 最小触发间隔（秒），避免重复触发

        Returns:
            如果检测到连续手势，返回手势类型；否则返回None
        """
        if len(self.history) < self.required_consecutive:
            return None

        # 检查最近N个手势是否相同
        recent_gestures = list(self.history)
        if len(recent_gestures) < self.required_consecutive:
            return None

        # 获取最近的N个手势
        recent_n = recent_gestures[-self.required_consecutive:]
        gesture_types = [g[0] for g in recent_n]

        # 检查所有手势是否相同
        first_gesture = gesture_types[0]
        if all(g == first_gesture for g in gesture_types):
            # 检查是否在最小触发间隔内
            if current_time - self.last_triggered_time < min_interval:
                return None

            # 检查是否与上次触发的手势相同
            if first_gesture == self.last_triggered_gesture:
                return None

            # 记录触发时间和手势
            self.last_triggered_gesture = first_gesture
            self.last_triggered_time = current_time

            # 不清空历史记录，但保留最新的一部分
            # 只清除触发的手势，保留其他手势
            self.history = deque(recent_gestures[-1:])  # 只保留最后一个手势

            return first_gesture

        return None

    def clear_history(self):
        """清空历史记录"""
        self.history.clear()
        self.last_triggered_gesture = None
        self.last_triggered_time = 0

    def get_history_status(self) -> str:
        """获取历史记录状态信息"""
        if not self.history:
            return "无手势记录"

        # 统计最近的手势
        recent_count = len(self.history)

        if recent_count >= self.required_consecutive:
            # 检查最近的手势是否一致
            recent_gestures = [g[0] for g in list(self.history)[-self.required_consecutive:]]
            if all(g == recent_gestures[0] for g in recent_gestures):
                return f"检测到连续{recent_count}个手势: {recent_gestures[0]}"
            else:
                return f"手势不一致: {recent_count}/{self.required_consecutive}"
        else:
            return f"手势记录: {recent_count}/{self.required_consecutive}"

    def get_last_triggered_gesture(self) -> Optional[str]:
        """获取上次触发的手势"""
        return self.last_triggered_gesture


def main():
    """游戏主函数"""
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption('贪吃蛇 - 手势控制版')

    font_small = pygame.font.SysFont('SimHei', 20)
    font_large = pygame.font.SysFont(None, 72)
    font_pause = pygame.font.SysFont(None, 48)
    font_info = pygame.font.SysFont('SimHei', 16)
    font_status = pygame.font.SysFont('SimHei', 18)  # 状态信息字体

    game_over_width, game_over_height = font_large.size('GAME OVER')

    game_over = True
    game_started = False
    game_paused = False

    control_mode = CONTROL_KEYBOARD  # 默认键盘控制
    gesture_confidence = 0.0
    last_gesture_time = 0
    last_gesture_type = None

    performance_monitor = PerformanceMonitor()

    gesture_queue = queue.Queue(maxsize=20)  # 增加队列大小，避免丢失手势

    # 初始化摄像头显示窗口
    camera_display = CameraDisplay("Camera Feed", width=640, height=480)

    # 初始化手势识别器
    gesture_recognizer = GestureRecognizer(gesture_queue, performance_monitor, camera_display)

    # 初始化手势历史记录器，时间窗口1秒，需要连续3个手势
    gesture_history = GestureHistory(window_size=1.0, required_consecutive=3)

    snake, food, food_style, direction, score = reset_game()
    next_direction = direction  # 存储下一帧的方向

    # 记录上一个手势指令
    last_gesture_command = None

    # 将基础速度调整为原来的0.8倍，使蛇移动得更慢
    base_speed = 0.3 * 1.25  # 调整为原来的0.8倍速度
    current_speed = base_speed
    last_move_time = None

    running = True
    while running:
        current_time = time.time()

        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
                gesture_recognizer.stop()
                camera_display.stop()

            elif event.type == KEYDOWN:
                if event.key == K_RETURN:
                    if game_over:
                        snake, food, food_style, direction, score = reset_game()
                        next_direction = direction
                        game_over = False
                        game_started = True
                        game_paused = False
                        current_speed = base_speed
                        last_move_time = current_time
                        last_gesture_command = None  # 重置上一个手势指令
                        print("游戏开始")

                        # 清空手势历史记录
                        gesture_history.clear_history()

                        # 重新开始时，如果当前是手势控制模式，重新启动手势识别线程和摄像头窗口
                        if control_mode == CONTROL_GESTURE:
                            print("重新启动手势识别线程和摄像头窗口")
                            # 启动摄像头显示窗口
                            camera_display.start()
                            if not gesture_recognizer.start():
                                print("手势识别启动失败，切换到键盘模拟模式")

                elif event.key == K_SPACE:
                    if game_started and not game_over:
                        game_paused = not game_paused
                        print(f"游戏暂停: {game_paused}")

                # G键切换控制模式
                elif event.key == K_g:
                    control_mode = CONTROL_GESTURE if control_mode == CONTROL_KEYBOARD else CONTROL_KEYBOARD
                    if control_mode == CONTROL_GESTURE:
                        print("切换到手势控制模式")
                        # 启动摄像头显示
                        camera_display.start()
                        if not gesture_recognizer.start():
                            print("手势识别启动失败，保持键盘控制模式")
                            control_mode = CONTROL_KEYBOARD
                            camera_display.stop()
                        else:
                            # 切换到手势控制模式时清空上一个手势指令
                            last_gesture_command = None
                    else:
                        print("切换到键盘控制模式")
                        gesture_recognizer.stop()
                        camera_display.stop()
                        # 切换到键盘控制模式时清空手势历史记录和上一个手势指令
                        gesture_history.clear_history()
                        last_gesture_command = None

                # 方向控制（仅键盘模式下有效）
                elif game_started and not game_over and not game_paused and control_mode == CONTROL_KEYBOARD:
                    if event.key in (K_UP, K_w):
                        if direction != (0, 1):
                            next_direction = (0, -1)
                    elif event.key in (K_DOWN, K_s):
                        if direction != (0, -1):
                            next_direction = (0, 1)
                    elif event.key in (K_LEFT, K_a):
                        if direction != (1, 0):
                            next_direction = (-1, 0)
                    elif event.key in (K_RIGHT, K_d):
                        if direction != (-1, 0):
                            next_direction = (1, 0)

        # 处理手势队列中的结果
        gesture_processed = False
        if control_mode == CONTROL_GESTURE and not game_over and game_started:
            try:
                while True:
                    gesture_result = gesture_queue.get_nowait()
                    direction_result, confidence, gesture_time, gesture_type = gesture_result

                    # 只处理1秒内的手势
                    if current_time - gesture_time < 1.0:
                        # 将手势添加到历史记录
                        if gesture_type:
                            gesture_history.add_gesture(gesture_type, gesture_time)

                        # 检查是否有连续手势
                        detected_gesture = gesture_history.check_consecutive_gesture(current_time, min_interval=0.3)

                        if detected_gesture:
                            print(f"检测到连续手势: {detected_gesture}，触发控制")

                            # 处理特殊手势
                            if detected_gesture == "wave":  # 暂停手势
                                if not game_over and game_started and not game_paused:
                                    # 检查是否与上一个指令相同
                                    if detected_gesture != last_gesture_command:
                                        game_paused = True
                                        last_gesture_command = detected_gesture
                                        print(f"连续手势暂停游戏")
                                        gesture_processed = True
                                    else:
                                        print(f"重复手势: {detected_gesture}，跳过")
                            elif detected_gesture == "forward":  # 开始/继续手势
                                if game_paused:  # 游戏暂停状态下，forward手势继续游戏
                                    if detected_gesture != last_gesture_command:
                                        game_paused = False
                                        last_gesture_command = detected_gesture
                                        print(f"连续手势继续游戏")
                                        gesture_processed = True
                                    else:
                                        print(f"重复手势: {detected_gesture}，跳过")
                                elif game_over:  # 游戏结束状态下，forward手势重新开始游戏
                                    if detected_gesture != last_gesture_command:
                                        snake, food, food_style, direction, score = reset_game()
                                        next_direction = direction
                                        game_over = False
                                        game_started = True
                                        game_paused = False
                                        current_speed = base_speed
                                        last_move_time = current_time
                                        last_gesture_command = detected_gesture
                                        print("连续手势开始游戏")
                                        gesture_processed = True
                                    else:
                                        print(f"重复手势: {detected_gesture}，跳过")
                                # 游戏运行状态下，forward手势不做任何操作
                            elif detected_gesture in ["up", "down", "left", "right"]:  # 方向手势
                                if not game_paused:  # 只有在游戏运行状态下才处理方向手势
                                    # 不能直接反向移动
                                    direction_vector = gesture_recognizer.gesture_to_direction[detected_gesture]
                                    if (direction_vector[0] != -direction[0] or
                                            direction_vector[1] != -direction[1]):
                                        # 检查是否与上一个指令相同
                                        if detected_gesture != last_gesture_command:
                                            next_direction = direction_vector
                                            gesture_confidence = confidence
                                            last_gesture_time = current_time
                                            last_gesture_type = detected_gesture
                                            last_gesture_command = detected_gesture
                                            gesture_processed = True
                                            print(f"连续手势控制: {detected_gesture}, 置信度: {confidence:.2f}")
                                        else:
                                            print(f"重复手势: {detected_gesture}，跳过")

                        # 更新最后的单次手势信息（用于显示）
                        if gesture_type and current_time - gesture_time < 1.0:
                            last_gesture_time = gesture_time
                            last_gesture_type = gesture_type
                            gesture_confidence = confidence

                    gesture_queue.task_done()
            except queue.Empty:
                pass

        screen.fill(BACKGROUND)

        draw_grid(screen)

        if not game_over and not game_paused and game_started:
            direction = next_direction

            if last_move_time is None or current_time - last_move_time > current_speed:
                last_move_time = current_time

                head_x, head_y = snake[0]
                next_x = head_x + direction[0]
                next_y = head_y + direction[1]
                next_position = (next_x, next_y)

                if next_position == food:
                    snake.appendleft(next_position)
                    score += food_style[0]

                    # 调整速度计算，确保速度比原来慢0.8倍
                    base_speed_adjusted = 0.3  # 原始基础速度
                    calculated_speed = max(0.05, base_speed_adjusted - 0.03 * (score // 100))
                    # 调整为原来的0.8倍速度（移动间隔增加1.25倍）
                    current_speed = calculated_speed * 1.25

                    food = create_food(snake)
                    food_style = get_random_food_style()

                    # 吃到食物后重置上一个手势指令，允许相同方向再次移动
                    if control_mode == CONTROL_GESTURE:
                        last_gesture_command = None
                        print("吃到食物，重置手势指令记录")
                else:
                    if (AREA_X[0] <= next_x <= AREA_X[1] and
                            AREA_Y[0] <= next_y <= AREA_Y[1] and
                            next_position not in snake):
                        snake.appendleft(next_position)
                        snake.pop()
                    else:
                        game_over = True
                        # 游戏结束时，只在手势控制模式下停止手势识别线程
                        # 但不要停止摄像头显示窗口，因为用户可能需要重新开始
                        if control_mode == CONTROL_GESTURE:
                            print("游戏结束，停止手势识别线程")
                            gesture_recognizer.stop()
                            # 注意：这里不停止摄像头显示窗口，让用户可以继续看到摄像头画面
                        print("游戏结束")

        if not game_over:
            draw_food(screen, food, food_style[1])
            draw_snake(screen, snake)

        control_mode_text = "手势控制" if control_mode == CONTROL_GESTURE else "键盘控制"
        control_color = GREEN if control_mode == CONTROL_GESTURE else WHITE

        print_text(screen, font_small, 30, 7, f'速度: {score // 100}')
        print_text(screen, font_small, 450, 7, f'得分: {score}')
        print_text(screen, font_small, 200, 7, f'控制模式: {control_mode_text}', control_color)

        # 显示当前速度
        speed_ratio = 0.8  # 当前速度为原始速度的0.8倍
        print_text(screen, font_small, 350, 7, f'速度比: {speed_ratio}', YELLOW)

        # 显示上一个手势指令
        if control_mode == CONTROL_GESTURE and last_gesture_command:
            print_text(screen, font_small, 500, 7, f'上指令: {last_gesture_command}', CYAN)

        # 绘制手势识别状态信息
        if control_mode == CONTROL_GESTURE:
            fps = performance_monitor.get_current_fps()
            avg_fps = performance_monitor.get_avg_fps()
            avg_latency = performance_monitor.get_avg_latency()

            print_text(screen, font_info, 10, 30, f'手势FPS: {fps:.1f}', BLUE)
            print_text(screen, font_info, 10, 50, f'平均FPS: {avg_fps:.1f}', BLUE)
            print_text(screen, font_info, 10, 70, f'平均延迟: {avg_latency:.1f}ms', BLUE)

            if gesture_confidence > 0:
                confidence_color = GREEN if gesture_confidence > 0.7 else YELLOW
                print_text(screen, font_info, 10, 90, f'置信度: {gesture_confidence:.2f}', confidence_color)

            # 显示手势识别状态
            y_offset = 110
            # 检查手势识别器是否有调试属性
            if hasattr(gesture_recognizer, 'last_gesture'):
                gesture_status = gesture_recognizer.last_gesture
            else:
                gesture_status = "未知"
            gesture_color = GREEN if gesture_recognizer.running else RED
            print_text(screen, font_status, 10, y_offset, f'手势状态: {gesture_status}', gesture_color)

            y_offset += 20
            if hasattr(gesture_recognizer, 'last_hand_detected'):
                hand_detected_color = GREEN if gesture_recognizer.last_hand_detected else RED
                hand_status = "检测到手部" if gesture_recognizer.last_hand_detected else "未检测到手部"
            else:
                hand_detected_color = RED
                hand_status = "线程未运行"
            print_text(screen, font_status, 10, y_offset, f'手部检测: {hand_status}', hand_detected_color)

            y_offset += 20
            if hasattr(gesture_recognizer, 'processing_time') and gesture_recognizer.running:
                print_text(screen, font_status, 10, y_offset, f'处理时间: {gesture_recognizer.processing_time:.1f}ms',
                           YELLOW)
            else:
                print_text(screen, font_status, 10, y_offset, '处理时间: 未运行', YELLOW)

            if last_gesture_type:
                y_offset += 20
                time_since_last = current_time - last_gesture_time
                time_color = GREEN if time_since_last < 1.0 else YELLOW
                print_text(screen, font_status, 10, y_offset, f'上次手势: {last_gesture_type}', time_color)

                y_offset += 20
                direction_text = f"控制方向: ({next_direction[0]}, {next_direction[1]})"
                print_text(screen, font_status, 10, y_offset, direction_text, MAGENTA)

            # 显示连续手势检测状态
            y_offset += 20
            history_status = gesture_history.get_history_status()
            print_text(screen, font_status, 10, y_offset, f'连续检测: {history_status}', ORANGE)

            # 显示检测时间窗口
            y_offset += 20
            window_text = f"时间窗口: 1秒内"
            print_text(screen, font_status, 10, y_offset, window_text, CYAN)

            # 显示上一个触发的手势指令
            y_offset += 20
            last_triggered = gesture_history.get_last_triggered_gesture()
            if last_triggered:
                trigger_text = f"上一个触发: {last_triggered}"
            else:
                trigger_text = "上一个触发: 无"
            print_text(screen, font_status, 10, y_offset, trigger_text, MAGENTA)

            # 添加线程状态显示
            y_offset += 20
            thread_status = "运行中" if gesture_recognizer.running else "已停止"
            thread_color = GREEN if thread_status == "运行中" else RED
            print_text(screen, font_status, 10, y_offset, f'识别线程: {thread_status}', thread_color)

            # 显示模型加载状态
            y_offset += 20
            model_status = "已加载" if hasattr(gesture_recognizer,
                                               'model_loaded') and gesture_recognizer.model_loaded else "未加载"
            model_color = GREEN if model_status == "已加载" else RED
            print_text(screen, font_status, 10, y_offset, f'模型状态: {model_status}', model_color)

            # 显示摄像头窗口状态
            y_offset += 20
            camera_status = "已开启" if camera_display.running else "已关闭"
            camera_color = GREEN if camera_display.running else RED
            print_text(screen, font_status, 10, y_offset, f'摄像头窗口: {camera_status}', camera_color)

        # 绘制控制提示
        print_text(screen, font_info, 10, SCREEN_HEIGHT - 100, '按G键切换控制模式', WHITE)
        print_text(screen, font_info, 10, SCREEN_HEIGHT - 80, '按回车开始/重新开始', WHITE)
        print_text(screen, font_info, 10, SCREEN_HEIGHT - 60, '按空格暂停/继续', WHITE)
        print_text(screen, font_info, 10, SCREEN_HEIGHT - 40, '方向键或WASD控制移动', WHITE)

        if control_mode == CONTROL_GESTURE:
            print_text(screen, font_info, SCREEN_WIDTH - 300, SCREEN_HEIGHT - 60,
                       '手势控制说明:', WHITE)
            print_text(screen, font_info, SCREEN_WIDTH - 300, SCREEN_HEIGHT - 40,
                       'wave: 暂停游戏', WHITE)
            print_text(screen, font_info, SCREEN_WIDTH - 300, SCREEN_HEIGHT - 20,
                       'forward: 继续游戏', WHITE)
            # 添加连续手势检测说明
            print_text(screen, font_info, SCREEN_WIDTH - 300, SCREEN_HEIGHT - 80,
                       '需要1秒内连续3个相同手势触发', WHITE)
            print_text(screen, font_info, SCREEN_WIDTH - 300, SCREEN_HEIGHT - 100,
                       '重复手势指令不会触发', YELLOW)

        if game_paused and not game_over:
            pause_width, pause_height = font_pause.size('游戏暂停')
            print_text(
                screen, font_pause,
                (SCREEN_WIDTH - pause_width) // 2,
                (SCREEN_HEIGHT - pause_height) // 2,
                '游戏暂停', RED
            )
            if control_mode == CONTROL_GESTURE:
                print_text(
                    screen, font_small,
                    (SCREEN_WIDTH - 200) // 2,
                    (SCREEN_HEIGHT - pause_height) // 2 + 60,
                    'forward手势继续', WHITE
                )
            else:
                print_text(
                    screen, font_small,
                    (SCREEN_WIDTH - 200) // 2,
                    (SCREEN_HEIGHT - pause_height) // 2 + 60,
                    '按空格键继续', WHITE
                )

        if game_over and game_started:
            print_text(
                screen, font_large,
                (SCREEN_WIDTH - game_over_width) // 2,
                (SCREEN_HEIGHT - game_over_height) // 2,
                'GAME OVER', RED
            )
            print_text(
                screen, font_small,
                (SCREEN_WIDTH - 200) // 2,
                (SCREEN_HEIGHT - game_over_height) // 2 + 80,
                '按回车键重新开始', WHITE
            )

        if not game_started:
            start_width, start_height = font_large.size('贪吃蛇 - 手势控制版')
            print_text(
                screen, font_large,
                (SCREEN_WIDTH - start_width) // 2,
                (SCREEN_HEIGHT - start_height) // 2 - 50,
                '贪吃蛇 - 手势控制版', RED
            )
            print_text(
                screen, font_small,
                (SCREEN_WIDTH - 200) // 2,
                (SCREEN_HEIGHT - start_height) // 2 + 20,
                '按回车键开始游戏', WHITE
            )
            if control_mode == CONTROL_GESTURE:
                print_text(
                    screen, font_small,
                    (SCREEN_WIDTH - 300) // 2,
                    (SCREEN_HEIGHT - start_height) // 2 + 60,
                    'forward手势开始，wave手势暂停', WHITE
                )
                print_text(
                    screen, font_small,
                    (SCREEN_WIDTH - 300) // 2,
                    (SCREEN_HEIGHT - start_height) // 2 + 80,
                    '需要1秒内连续3个相同手势触发', WHITE
                )
                print_text(
                    screen, font_small,
                    (SCREEN_WIDTH - 300) // 2,
                    (SCREEN_HEIGHT - start_height) // 2 + 100,
                    '重复手势指令不会触发', YELLOW
                )
                print_text(
                    screen, font_small,
                    (SCREEN_WIDTH - 300) // 2,
                    (SCREEN_HEIGHT - start_height) // 2 + 120,
                    '当前速度: 原速度的0.8倍', YELLOW
                )
            else:
                print_text(
                    screen, font_small,
                    (SCREEN_WIDTH - 300) // 2,
                    (SCREEN_HEIGHT - start_height) // 2 + 60,
                    '按G键切换控制模式，空格键暂停', WHITE
                )

        pygame.display.update()

    # 清理资源
    gesture_recognizer.stop()
    camera_display.stop()
    pygame.quit()
    cv2.destroyAllWindows()  # 确保关闭所有OpenCV窗口
    sys.exit()


if __name__ == '__main__':
    main()