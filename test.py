import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, \
    f1_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import pandas as pd
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

from mobilenetV2 import MobileNetV2


class ModelTester:
    def __init__(self, model_weights_path, class_indices_path, test_data_root,
                 device=None, batch_size=16):
        """
        初始化模型测试器

        Args:
            model_weights_path: 模型权重路径
            class_indices_path: 类别索引文件路径
            test_data_root: 测试数据根目录
            device: 计算设备
            batch_size: 批量大小
        """
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.test_data_root = test_data_root

        # 检查文件是否存在
        assert os.path.exists(model_weights_path), f"模型权重文件不存在: {model_weights_path}"
        assert os.path.exists(class_indices_path), f"类别索引文件不存在: {class_indices_path}"
        assert os.path.exists(test_data_root), f"测试数据目录不存在: {test_data_root}"

        # 加载类别索引
        with open(class_indices_path, "r") as f:
            data = json.load(f)  # {"0": "down", "1": "left", ...}
            # 创建索引到类别的映射
            self.idx_to_class = {int(k): v for k, v in data.items()}
            # 创建类别到索引的映射
            self.class_indict = {v: k for k, v in self.idx_to_class.items()}
            self.class_names = [self.idx_to_class[i] for i in range(len(self.idx_to_class))]
            # 保存原始类别索引
            self.class_indices = data  # 添加这行，保存原始索引

        print(f"设备: {self.device}")
        self.num_classes = 6
        print(f"类别数量: {self.num_classes}")
        print(f"类别名称: {self.class_names}")

        # 创建模型
        self.model = MobileNetV2(num_classes=self.num_classes).to(self.device)

        # 加载模型权重
        checkpoint = torch.load(model_weights_path, map_location=self.device)

        # 检查是否是完整的模型保存（包含结构信息）
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # 加载完整模型
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print("✓ 加载完整模型（包含结构和权重）")
        else:
            # 只加载权重
            self.model.load_state_dict(checkpoint)
            print("✓ 加载模型权重")

        self.model.eval()

        # 创建测试数据转换
        self.data_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # 加载测试数据集
        self.test_dataset = datasets.ImageFolder(
            root=os.path.join(self.test_data_root, "Test"),
            transform=self.data_transform
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=min(4, os.cpu_count())
        )

        self.test_num = len(self.test_dataset)
        print(f"测试集样本数量: {self.test_num}")

    def evaluate(self, save_dir='sixclass/test_results'):
        """
        在测试集上评估模型
        """
        os.makedirs(save_dir, exist_ok=True)

        print(f"\n{'=' * 60}")
        print("开始在测试集上评估模型")
        print(f"{'=' * 60}")

        all_predictions = []
        all_labels = []
        all_probs = []

        # 禁用梯度计算
        with torch.no_grad():
            test_bar = tqdm(self.test_loader, desc="测试中", file=sys.stdout)
            for images, labels in test_bar:
                images = images.to(self.device)
                labels = labels.to(self.device)

                # 前向传播
                outputs = self.model(images)

                # 获取预测结果
                _, predictions = torch.max(outputs, 1)

                # 计算softmax概率
                probs = torch.nn.functional.softmax(outputs, dim=1)

                # 保存结果
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        # 转换为numpy数组
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        # 计算评估指标
        self._compute_metrics(all_labels, all_predictions, all_probs, save_dir)

        # 保存详细预测结果
        self._save_predictions(all_labels, all_predictions, all_probs, save_dir)

    def _compute_metrics(self, true_labels, pred_labels, probs, save_dir):
        """
        计算并保存所有评估指标
        """
        print(f"\n{'=' * 60}")
        print("计算评估指标")
        print(f"{'=' * 60}")

        # 基础指标
        accuracy = accuracy_score(true_labels, pred_labels)
        macro_precision = precision_score(true_labels, pred_labels, average='macro', zero_division=0)
        macro_recall = recall_score(true_labels, pred_labels, average='macro', zero_division=0)
        macro_f1 = f1_score(true_labels, pred_labels, average='macro', zero_division=0)

        # 每个类别的指标
        class_precision = precision_score(true_labels, pred_labels, average=None, zero_division=0)
        class_recall = recall_score(true_labels, pred_labels, average=None, zero_division=0)
        class_f1 = f1_score(true_labels, pred_labels, average=None, zero_division=0)

        # 生成分类报告
        report = classification_report(true_labels, pred_labels,
                                       target_names=self.class_names,
                                       digits=4,
                                       zero_division=0)

        # 计算混淆矩阵
        cm = confusion_matrix(true_labels, pred_labels)

        # 计算归一化混淆矩阵（按行归一化）
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)  # 处理除零情况

        # 保存结果
        self._save_metrics_to_file(accuracy, macro_precision, macro_recall, macro_f1,
                                   class_precision, class_recall, class_f1, report, save_dir)

        # 生成可视化
        self._plot_confusion_matrix(cm, "混淆矩阵", save_dir)
        self._plot_confusion_matrix(cm_normalized, "归一化混淆矩阵", save_dir, normalize=True)
        self._plot_metrics_by_class(class_precision, class_recall, class_f1, save_dir)

        # 打印总结
        self._print_summary(accuracy, macro_precision, macro_recall, macro_f1, report)

    def _save_metrics_to_file(self, accuracy, macro_precision, macro_recall, macro_f1,
                              class_precision, class_recall, class_f1, report, save_dir):
        """
        保存指标到文件
        """
        # 保存详细报告
        with open(os.path.join(save_dir, 'classification_report.txt'), 'w', encoding='utf-8') as f:
            f.write("分类报告\n")
            f.write("=" * 50 + "\n\n")
            f.write(report)
            f.write("\n\n")

            f.write("总体指标\n")
            f.write("-" * 30 + "\n")
            f.write(f"准确率 (Accuracy): {accuracy:.4f}\n")
            f.write(f"宏平均精确率 (Macro Precision): {macro_precision:.4f}\n")
            f.write(f"宏平均召回率 (Macro Recall): {macro_recall:.4f}\n")
            f.write(f"宏平均F1分数 (Macro F1): {macro_f1:.4f}\n")
            f.write("\n")

            f.write("各类别指标\n")
            f.write("-" * 30 + "\n")
            for i, class_name in enumerate(self.class_names):
                f.write(f"{class_name}:\n")
                f.write(f"  精确率: {class_precision[i]:.4f}\n")
                f.write(f"  召回率: {class_recall[i]:.4f}\n")
                f.write(f"  F1分数: {class_f1[i]:.4f}\n")

        # 保存为CSV格式
        metrics_df = pd.DataFrame({
            '类别': self.class_names,
            '精确率': class_precision,
            '召回率': class_recall,
            'F1分数': class_f1
        })
        metrics_df.to_csv(os.path.join(save_dir, 'class_metrics.csv'), index=False, encoding='utf-8')

        # 保存总体指标
        overall_metrics = pd.DataFrame({
            '指标': ['准确率', '宏平均精确率', '宏平均召回率', '宏平均F1分数'],
            '值': [accuracy, macro_precision, macro_recall, macro_f1]
        })
        overall_metrics.to_csv(os.path.join(save_dir, 'overall_metrics.csv'), index=False, encoding='utf-8')

        print(f"✓ 指标已保存到: {save_dir}")

    def _plot_confusion_matrix(self, cm, title, save_dir, normalize=False):
        """
        绘制混淆矩阵
        """
        plt.figure(figsize=(12, 10))

        if normalize:
            vmin, vmax = 0, 1
            fmt = '.2f'
            cbar_label = '比例'
        else:
            vmin, vmax = None, None
            fmt = 'd'
            cbar_label = '数量'

        # 创建热力图
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names,
                    vmin=vmin, vmax=vmax,
                    cbar_kws={'label': cbar_label})

        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('预测标签', fontsize=12)
        plt.ylabel('真实标签', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

        plt.tight_layout()

        # 保存图片
        filename = 'normalized_confusion_matrix.png' if normalize else 'confusion_matrix.png'
        plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ {title} 已保存")

    def _plot_metrics_by_class(self, precision, recall, f1, save_dir):
        """
        绘制各类别指标柱状图
        """
        x = np.arange(len(self.class_names))
        width = 0.25

        plt.figure(figsize=(14, 6))

        plt.bar(x - width, precision, width, label='精确率', alpha=0.8, color='skyblue')
        plt.bar(x, recall, width, label='召回率', alpha=0.8, color='lightgreen')
        plt.bar(x + width, f1, width, label='F1分数', alpha=0.8, color='salmon')

        plt.xlabel('类别', fontsize=12)
        plt.ylabel('分数', fontsize=12)
        plt.title('各类别评估指标', fontsize=16, fontweight='bold')
        plt.xticks(x, self.class_names, rotation=45, ha='right')
        plt.ylim([0, 1.05])
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3, axis='y')

        # 在每个柱子上添加数值
        for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
            plt.text(i - width, p + 0.01, f'{p:.2f}', ha='center', va='bottom', fontsize=9)
            plt.text(i, r + 0.01, f'{r:.2f}', ha='center', va='bottom', fontsize=9)
            plt.text(i + width, f + 0.01, f'{f:.2f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'class_metrics_bar.png'), dpi=300, bbox_inches='tight')
        plt.close()

        print("✓ 各类别指标柱状图已保存")

    def _save_predictions(self, true_labels, pred_labels, probs, save_dir):
        """
        保存每个样本的详细预测结果
        """
        # 获取测试集的图片路径
        image_paths = [os.path.join(self.test_dataset.root, self.test_dataset.imgs[i][0])
                       for i in range(len(self.test_dataset))]

        # 获取类别标签
        image_classes = [self.idx_to_class[true_labels[i]] for i in range(len(true_labels))]
        predicted_classes = [self.idx_to_class[pred_labels[i]] for i in range(len(pred_labels))]

        # 获取每个类别的置信度
        confidence_scores = []
        for i in range(len(probs)):
            confidence_scores.append(probs[i][pred_labels[i]])

        # 获取top-3预测结果
        top3_predictions = []
        top3_confidences = []

        for prob in probs:
            # 获取概率最大的3个类别
            top3_idx = np.argsort(prob)[-3:][::-1]
            top3_classes = [self.idx_to_class[idx] for idx in top3_idx]
            top3_probs = [prob[idx] for idx in top3_idx]

            top3_predictions.append(', '.join(top3_classes))
            top3_confidences.append(', '.join([f'{p:.4f}' for p in top3_probs]))

        # 创建结果DataFrame
        results_df = pd.DataFrame({
            'image_path': image_paths,
            'true_label': image_classes,
            'predicted_label': predicted_classes,
            'is_correct': (true_labels == pred_labels).astype(int),
            'confidence': confidence_scores,
            'top3_predictions': top3_predictions,
            'top3_confidences': top3_confidences
        })

        # 添加每个类别的概率
        for i, class_name in enumerate(self.class_names):
            results_df[f'prob_{class_name}'] = probs[:, i]

        # 保存结果
        results_df.to_csv(os.path.join(save_dir, 'detailed_predictions.csv'), index=False, encoding='utf-8')

        # 保存错误预测样本
        wrong_predictions = results_df[results_df['is_correct'] == 0]
        wrong_predictions.to_csv(os.path.join(save_dir, 'wrong_predictions.csv'),
                                 index=False, encoding='utf-8')

        print(f"✓ 详细预测结果已保存，共{len(results_df)}个样本")
        print(f"✓ 错误预测样本: {len(wrong_predictions)}个，已单独保存")

    def _print_summary(self, accuracy, macro_precision, macro_recall, macro_f1, report):
        """
        打印评估总结
        """
        print(f"\n{'=' * 60}")
        print("评估总结")
        print(f"{'=' * 60}")
        print(f"总体准确率: {accuracy:.4f}")
        print(f"宏平均精确率: {macro_precision:.4f}")
        print(f"宏平均召回率: {macro_recall:.4f}")
        print(f"宏平均F1分数: {macro_f1:.4f}")
        print(f"\n分类报告:\n{report}")

    def generate_error_analysis_report(self, save_dir='sixclass/test_results'):
        """
        生成错误分析报告
        """
        error_csv_path = os.path.join(save_dir, 'wrong_predictions.csv')
        if not os.path.exists(error_csv_path):
            print("错误预测文件不存在，请先运行evaluate()方法")
            return

        # 读取错误预测
        error_df = pd.read_csv(error_csv_path)

        if len(error_df) == 0:
            print("没有错误预测样本")
            return

        # 统计各类别的错误情况
        error_analysis = {}

        # 首先，统计测试集中每个类别的总样本数
        class_counts = {}
        for i, class_name in enumerate(self.class_names):
            class_idx = i  # 因为类别索引是从0开始的整数
            class_counts[class_name] = sum(1 for label in self.test_dataset.targets if label == class_idx)

        for class_name in self.class_names:
            class_errors = error_df[error_df['true_label'] == class_name]
            if len(class_errors) > 0:
                error_count = len(class_errors)
                most_common_error = class_errors['predicted_label'].mode()[0] if len(class_errors) > 0 else 'N/A'

                # 计算错误率
                total_class_samples = class_counts.get(class_name, 0)
                error_rate = error_count / total_class_samples if total_class_samples > 0 else 0

                error_analysis[class_name] = {
                    'error_count': error_count,
                    'most_common_error': most_common_error,
                    'error_rate': error_rate
                }

        # 保存错误分析报告
        with open(os.path.join(save_dir, 'error_analysis.txt'), 'w', encoding='utf-8') as f:
            f.write("错误分析报告\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"总错误样本数: {len(error_df)} / {self.test_num} ")
            f.write(f"({len(error_df) / self.test_num * 100:.2f}%)\n\n")

            f.write("各类别错误分析:\n")
            f.write("-" * 40 + "\n")
            for class_name, analysis in error_analysis.items():
                f.write(f"{class_name}:\n")
                f.write(f"  错误数: {analysis['error_count']}\n")
                f.write(f"  最常见错误类别: {analysis['most_common_error']}\n")
                f.write(f"  错误率: {analysis['error_rate'] * 100:.2f}%\n")
                f.write(f"  总样本数: {class_counts.get(class_name, 0)}\n\n")

        # 创建错误分析的可视化
        self._plot_error_analysis(error_analysis, class_counts, save_dir)

        print(f"✓ 错误分析报告已保存到: {save_dir}")

    def _plot_error_analysis(self, error_analysis, class_counts, save_dir):
        """
        绘制错误分析图
        """
        if not error_analysis:
            return

        # 准备数据
        classes = list(error_analysis.keys())
        error_counts = [error_analysis[c]['error_count'] for c in classes]
        total_counts = [class_counts.get(c, 0) for c in classes]
        error_rates = [error_analysis[c]['error_rate'] * 100 for c in classes]  # 转换为百分比

        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. 错误数量柱状图
        axes[0, 0].bar(classes, error_counts, color='salmon', alpha=0.8)
        axes[0, 0].set_title('各类别错误数量', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('类别', fontsize=12)
        axes[0, 0].set_ylabel('错误数量', fontsize=12)
        axes[0, 0].tick_params(axis='x', rotation=45)

        # 在柱子上添加数值
        for i, v in enumerate(error_counts):
            axes[0, 0].text(i, v + 0.5, str(v), ha='center', va='bottom', fontsize=10)

        # 2. 错误率柱状图
        bars = axes[0, 1].bar(classes, error_rates, color='lightcoral', alpha=0.8)
        axes[0, 1].set_title('各类别错误率（百分比）', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('类别', fontsize=12)
        axes[0, 1].set_ylabel('错误率 (%)', fontsize=12)
        axes[0, 1].tick_params(axis='x', rotation=45)

        # 在柱子上添加数值
        for i, v in enumerate(error_rates):
            axes[0, 1].text(i, v + 0.5, f'{v:.1f}%', ha='center', va='bottom', fontsize=10)

        # 3. 错误样本与总样本对比
        x = np.arange(len(classes))
        width = 0.35

        axes[1, 0].bar(x - width / 2, total_counts, width, label='总样本数', color='lightblue', alpha=0.8)
        axes[1, 0].bar(x + width / 2, error_counts, width, label='错误样本数', color='salmon', alpha=0.8)
        axes[1, 0].set_title('各类别样本数量对比', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('类别', fontsize=12)
        axes[1, 0].set_ylabel('样本数量', fontsize=12)
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(classes, rotation=45)
        axes[1, 0].legend()

        # 4. 错误类别分布饼图
        if error_counts:
            axes[1, 1].pie(error_counts, labels=classes, autopct='%1.1f%%', startangle=90,
                           colors=plt.cm.Set3(np.linspace(0, 1, len(classes))))
            axes[1, 1].set_title('错误样本类别分布', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'error_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()

        print("✓ 错误分析图已保存")


def main():
    """
    主函数
    """
    import argparse

    parser = argparse.ArgumentParser(description='模型测试脚本')
    parser.add_argument('--model_path', type=str, default='sixclass/MobileNetV2_best.pth',
                        help='模型权重路径，默认为sixclass/MobileNetV2_best.pth')
    parser.add_argument('--class_indices', type=str, default='sixclass/class_indices.json',
                        help='类别索引文件路径，默认为sixclass/class_indices.json')
    parser.add_argument('--data_root', type=str, default='HandNavigation',
                        help='数据根目录，默认为HandNavigation')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批量大小，默认为16')
    parser.add_argument('--save_dir', type=str, default='sixclass/test_results',
                        help='结果保存目录，默认为sixclass/test_results')

    args = parser.parse_args()

    print("模型测试脚本")
    print("=" * 60)
    print(f"模型路径: {args.model_path}")
    print(f"类别索引: {args.class_indices}")
    print(f"数据目录: {args.data_root}")
    print(f"批量大小: {args.batch_size}")
    print(f"保存目录: {args.save_dir}")
    print("=" * 60)

    # 创建测试器
    try:
        tester = ModelTester(
            model_weights_path=args.model_path,
            class_indices_path=args.class_indices,
            test_data_root=args.data_root,
            batch_size=args.batch_size
        )

        # 评估模型
        tester.evaluate(save_dir=args.save_dir)

        # 生成错误分析报告
        tester.generate_error_analysis_report(save_dir=args.save_dir)

    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()