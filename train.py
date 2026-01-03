import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm

from mobilenetV2 import MobileNetV2

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# ==================== 新增：检查点保存和加载函数 ====================
def save_checkpoint(state, is_best=False, checkpoint_dir='checkpoints',
                    epoch=None, save_interval=10):
    """
    保存训练检查点

    Args:
        state: 包含模型状态、优化器状态等的字典
        is_best: 是否是最佳模型
        checkpoint_dir: 检查点保存目录
        epoch: 当前epoch
        save_interval: 每隔多少epoch保存一次
    """
    # 创建保存目录
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 如果是定期保存
    if epoch is not None and epoch % save_interval == 0:
        # 保存定期检查点
        regular_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch:03d}.pth')
        torch.save(state, regular_path)
        print(f"  -> Regular checkpoint saved: {regular_path}")

    # 保存最新的检查点
    latest_path = os.path.join(checkpoint_dir, 'checkpoint_latest.pth')
    torch.save(state, latest_path)
    print(f"  -> Latest checkpoint saved: {latest_path}")

    # 如果是当前最佳模型，额外保存
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'model_best.pth')
        torch.save(state, best_path)
        print(f"  -> Best checkpoint saved: {best_path}")


def create_checkpoint_state(model, optimizer, scheduler, epoch,
                            best_acc, train_losses, val_accuracies,
                            class_indices, config, is_finetune_phase=False):
    """
    创建检查点状态字典

    Args:
        model: 模型
        optimizer: 优化器
        scheduler: 学习率调度器
        epoch: 当前训练轮数
        best_acc: 当前最佳准确率
        train_losses: 训练损失列表
        val_accuracies: 验证准确率列表
        class_indices: 类别索引
        config: 训练配置
        is_finetune_phase: 是否在微调阶段
    """
    return {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_acc': best_acc,
        'train_losses': train_losses,
        'val_accuracies': val_accuracies,
        'class_indices': class_indices,
        'config': config,
        'is_finetune_phase': is_finetune_phase,
        'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }


def save_model_weights(model, save_path, additional_info=None):
    """
    只保存模型权重（兼容原始代码）

    Args:
        model: 模型
        save_path: 保存路径
        additional_info: 附加信息
    """
    # 确保保存目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 只保存模型权重
    torch.save(model.state_dict(), save_path)
    print(f"  -> Model weights saved: {save_path}")


def save_complete_model(model, save_path, class_indices=None, config=None):
    """
    保存完整的模型（包括结构和配置）

    Args:
        model: 模型
        save_path: 保存路径
        class_indices: 类别索引
        config: 训练配置
    """
    # 确保保存目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 保存完整的模型信息
    model_info = {
        'model_state_dict': model.state_dict(),
        'model_class_name': model.__class__.__name__,
        'num_classes': getattr(model, 'num_classes', None),
        'class_indices': class_indices,
        'config': config,
        'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    torch.save(model_info, save_path)
    print(f"  -> Complete model saved: {save_path}")


# ==================== 原有绘图和保存函数 ====================
def plot_training_curves(train_losses, val_accuracies, save_path='training_curves.png'):
    """绘制训练损失和验证准确率曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # 绘制训练损失曲线
    ax1.plot(train_losses, label='Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Curve')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 绘制验证准确率曲线
    ax2.plot(val_accuracies, label='Validation Accuracy', color='orange')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Validation Accuracy Curve')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {save_path}")


def plot_confusion_matrix(true_labels, pred_labels, class_names, save_path='confusion_matrix.png'):
    """绘制混淆矩阵"""
    cm = confusion_matrix(true_labels, pred_labels)

    plt.figure(figsize=(10, 8))

    # 使用seaborn绘制热力图
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)

    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 打印分类报告
    print("\nClassification Report:")
    print(classification_report(true_labels, pred_labels, target_names=class_names))

    print(f"Confusion matrix saved to {save_path}")


def save_training_results(train_losses, val_accuracies, best_epoch, best_acc, save_path='training_results.csv'):
    """保存训练结果到CSV文件"""
    results_df = pd.DataFrame({
        'epoch': list(range(1, len(train_losses) + 1)),
        'train_loss': train_losses,
        'val_accuracy': val_accuracies
    })

    # 添加最佳epoch标记
    results_df['is_best'] = results_df['epoch'] == best_epoch

    results_df.to_csv(save_path, index=False)

    # 创建汇总信息
    summary = {
        'best_epoch': int(best_epoch),
        'best_accuracy': float(best_acc),
        'final_epoch': len(train_losses),
        'final_accuracy': float(val_accuracies[-1]),
        'final_loss': float(train_losses[-1])
    }

    with open('sixclass/training_summary.json', 'w') as f:
        json.dump(summary, f, indent=4)

    print(f"Training results saved to {save_path}")
    print(f"Training summary saved to training_summary.json")
    print(f"\nTraining Summary:")
    print(f"  Best epoch: {best_epoch}")
    print(f"  Best accuracy: {best_acc:.4f}")
    print(f"  Final accuracy: {val_accuracies[-1]:.4f}")
    print(f"  Final loss: {train_losses[-1]:.4f}")


# ==================== 主训练函数 ====================
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device.")

    # 设置检查点保存参数
    checkpoint_dir = 'sixclass/checkpoints'
    checkpoint_save_interval = 10  # 每10个epoch保存一次检查点

    # 设置随机种子以保证可重复性
    torch.manual_seed(42)
    np.random.seed(42)

    batch_size = 16
    total_epochs = 55
    freeze_epochs = 25  # 第一阶段：冻结特征层训练轮数
    finetune_epochs = total_epochs - freeze_epochs  # 第二阶段：全网络微调轮数

    # 创建训练配置
    config = {
        'batch_size': batch_size,
        'total_epochs': total_epochs,
        'freeze_epochs': freeze_epochs,
        'finetune_epochs': finetune_epochs,
        'checkpoint_save_interval': checkpoint_save_interval,
        'checkpoint_dir': checkpoint_dir,
        'device': str(device),
        'model': 'MobileNetV2',
        'random_seed': 42
    }

    # 数据增强和预处理
    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),

            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # 加载数据集
    data_root = os.path.abspath(os.path.join(os.getcwd()))
    image_path = os.path.join(data_root, "HandNavigation")
    assert os.path.exists(image_path), f"{image_path} path does not exist."

    train_dataset = datasets.ImageFolder(
        root=os.path.join(image_path, "Train"),
        transform=data_transform["train"]
    )
    train_num = len(train_dataset)

    validate_dataset = datasets.ImageFolder(
        root=os.path.join(image_path, "Validation"),
        transform=data_transform["val"]
    )
    val_num = len(validate_dataset)

    # 保存类别索引映射
    class_to_idx = train_dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    json_str = json.dumps(idx_to_class, indent=4)
    with open('sixclass/class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    print(f"Classes: {class_names}")
    print(f"Class indices: {idx_to_class}")

    # 创建数据加载器
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print(f'Using {nw} dataloader workers every process')

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=nw
    )
    validate_loader = torch.utils.data.DataLoader(
        validate_dataset, batch_size=batch_size, shuffle=False, num_workers=nw
    )

    print(f"Using {train_num} images for training, {val_num} images for validation.")

    # 创建模型
    num_classes = len(class_names)
    net = MobileNetV2(num_classes=num_classes)

    # 加载预训练权重
    model_weight_path = "./mobilenet_v2-pre.pth"
    assert os.path.exists(model_weight_path), f"file {model_weight_path} does not exist."

    pre_weights = torch.load(model_weight_path, map_location='cpu')

    # 删除分类器权重（因为类别数不同）
    pre_dict = {k: v for k, v in pre_weights.items() if net.state_dict()[k].numel() == v.numel()}
    missing_keys, unexpected_keys = net.load_state_dict(pre_dict, strict=False)

    print(f"Missing keys: {missing_keys}")
    print(f"Unexpected keys: {unexpected_keys}")

    net.to(device)

    # 定义损失函数
    loss_function = nn.CrossEntropyLoss()

    # 用于记录训练过程
    train_losses = []
    val_accuracies = []
    best_acc = 0.0
    best_epoch = 0

    # 模型保存路径
    best_model_path = 'sixclass/MobileNetV2_best.pth'
    final_model_path = 'sixclass/MobileNetV2_final.pth'
    complete_model_path = 'sixclass/MobileNetV2_complete.pth'

    # ========== 第一阶段：冻结特征层，只训练分类层 ==========
    print("\n" + "=" * 60)
    print("Phase 1: Freeze backbone, train only classifier")
    print("=" * 60)

    # 冻结特征层
    for param in net.features.parameters():
        param.requires_grad = False

    # 只训练分类层
    params_to_train = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params_to_train, lr=0.0001)

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    for epoch in range(freeze_epochs):
        # 训练
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout, desc=f"Freeze Phase Epoch [{epoch + 1}/{freeze_epochs}]")

        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        # 验证
        net.eval()
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout, desc="Validating")
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accurate = acc / val_num
        avg_train_loss = running_loss / len(train_loader)

        # 记录结果
        train_losses.append(avg_train_loss)
        val_accuracies.append(val_accurate)

        # 更新学习率
        scheduler.step()

        print(
            f'[Freeze Phase Epoch {epoch + 1}] Train Loss: {avg_train_loss:.4f}, Val Accuracy: {val_accurate:.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')

        # 检查是否为最佳模型
        is_best = val_accurate > best_acc
        if is_best:
            best_acc = val_accurate
            best_epoch = epoch + 1
            print(f"  -> New best model! Accuracy: {best_acc:.4f}")

        # 保存模型权重（兼容原始代码）
        if is_best:
            save_model_weights(net, best_model_path)

        # 创建检查点状态
        checkpoint_state = create_checkpoint_state(
            model=net,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_acc=best_acc,
            train_losses=train_losses,
            val_accuracies=val_accuracies,
            class_indices=idx_to_class,
            config=config,
            is_finetune_phase=False
        )

        # 保存检查点
        save_checkpoint(
            state=checkpoint_state,
            is_best=is_best,
            checkpoint_dir=checkpoint_dir,
            epoch=epoch + 1,
            save_interval=checkpoint_save_interval
        )

    # ========== 第二阶段：解冻所有层，全网络微调 ==========
    print("\n" + "=" * 60)
    print("Phase 2: Unfreeze all layers, fine-tune whole network")
    print("=" * 60)

    # 解冻所有层
    for param in net.parameters():
        param.requires_grad = True

    # 使用更小的学习率进行微调
    optimizer = optim.Adam(net.parameters(), lr=0.00001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    for epoch in range(freeze_epochs, total_epochs):
        # 训练
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout, desc=f"Fine-tune Phase Epoch [{epoch + 1}/{total_epochs}]")

        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        # 验证
        net.eval()
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout, desc="Validating")
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accurate = acc / val_num
        avg_train_loss = running_loss / len(train_loader)

        # 记录结果
        train_losses.append(avg_train_loss)
        val_accuracies.append(val_accurate)

        # 更新学习率
        scheduler.step()

        print(
            f'[Fine-tune Phase Epoch {epoch + 1}] Train Loss: {avg_train_loss:.4f}, Val Accuracy: {val_accurate:.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')

        # 检查是否为最佳模型
        is_best = val_accurate > best_acc
        if is_best:
            best_acc = val_accurate
            best_epoch = epoch + 1
            print(f"  -> New best model! Accuracy: {best_acc:.4f}")

        # 保存模型权重（兼容原始代码）
        if is_best:
            save_model_weights(net, best_model_path)

        # 创建检查点状态
        checkpoint_state = create_checkpoint_state(
            model=net,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_acc=best_acc,
            train_losses=train_losses,
            val_accuracies=val_accuracies,
            class_indices=idx_to_class,
            config=config,
            is_finetune_phase=True
        )

        # 保存检查点
        save_checkpoint(
            state=checkpoint_state,
            is_best=is_best,
            checkpoint_dir=checkpoint_dir,
            epoch=epoch + 1,
            save_interval=checkpoint_save_interval
        )

    # 训练结束后保存最终模型
    print("\n" + "=" * 60)
    print("Saving final models")
    print("=" * 60)

    # 1. 只保存模型权重（兼容原始代码）
    save_model_weights(net, final_model_path)

    # 2. 保存完整模型（包括结构和配置）
    save_complete_model(
        model=net,
        save_path=complete_model_path,
        class_indices=idx_to_class,
        config=config
    )

    # 3. 保存最终检查点
    checkpoint_state = create_checkpoint_state(
        model=net,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=total_epochs - 1,
        best_acc=best_acc,
        train_losses=train_losses,
        val_accuracies=val_accuracies,
        class_indices=idx_to_class,
        config=config,
        is_finetune_phase=True
    )
    save_checkpoint(checkpoint_state, is_best=False, checkpoint_dir=checkpoint_dir)

    # 保存训练结果
    save_training_results(train_losses, val_accuracies, best_epoch, best_acc)

    # 绘制训练曲线
    plot_training_curves(train_losses, val_accuracies)

    # ========== 加载最佳模型并生成混淆矩阵 ==========
    print("\n" + "=" * 60)
    print("Generating confusion matrix with best model")
    print("=" * 60)

    # 加载最佳模型
    net_best = MobileNetV2(num_classes=num_classes)
    net_best.load_state_dict(torch.load(best_model_path, map_location=device))
    net_best.to(device)
    net_best.eval()

    # 收集所有预测和真实标签
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for val_data in tqdm(validate_loader, desc="Generating predictions"):
            val_images, val_labels = val_data
            outputs = net_best(val_images.to(device))
            _, predictions = torch.max(outputs, 1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(val_labels.numpy())

    # 绘制混淆矩阵
    plot_confusion_matrix(all_labels, all_predictions, class_names)

    # 计算并保存每个类别的准确率
    class_correct = {class_name: 0 for class_name in class_names}
    class_total = {class_name: 0 for class_name in class_names}

    for true_label, pred_label in zip(all_labels, all_predictions):
        class_name = idx_to_class[true_label]
        class_total[class_name] += 1
        if true_label == pred_label:
            class_correct[class_name] += 1

    class_accuracy = {}
    for class_name in class_names:
        if class_total[class_name] > 0:
            accuracy = class_correct[class_name] / class_total[class_name]
            class_accuracy[class_name] = accuracy

    # 保存每个类别的准确率
    accuracy_df = pd.DataFrame({
        'class': list(class_accuracy.keys()),
        'accuracy': list(class_accuracy.values()),
        'samples': [class_total[c] for c in class_accuracy.keys()]
    })
    accuracy_df.to_csv('sixclass/class_accuracy.csv', index=False)

    print("\nClass-wise Accuracy:")
    print(accuracy_df.to_string(index=False))

    # 保存所有检查点信息
    save_checkpoint_info(checkpoint_dir, best_epoch, best_acc)

    print('\n' + '=' * 60)
    print('Finished Training and Evaluation!')
    print(f'Best model saved to: {best_model_path}')
    print(f'Final model saved to: {final_model_path}')
    print(f'Complete model saved to: {complete_model_path}')
    print(f'All checkpoints saved to: {checkpoint_dir}')
    print(f'Best accuracy: {best_acc:.4f} at epoch {best_epoch}')
    print('=' * 60)


# ==================== 新增：检查点信息汇总函数 ====================
def save_checkpoint_info(checkpoint_dir, best_epoch, best_acc):
    """保存检查点信息汇总"""
    info = {
        'checkpoint_dir': checkpoint_dir,
        'best_epoch': int(best_epoch),
        'best_accuracy': float(best_acc),
        'created_time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'checkpoints': []
    }

    # 扫描检查点目录
    if os.path.exists(checkpoint_dir):
        for file in os.listdir(checkpoint_dir):
            if file.endswith('.pth'):
                file_path = os.path.join(checkpoint_dir, file)
                file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                file_time = datetime.datetime.fromtimestamp(
                    os.path.getmtime(file_path)).strftime('%Y-%m-%d %H:%M:%S')

                info['checkpoints'].append({
                    'name': file,
                    'size_mb': round(file_size, 2),
                    'modified_time': file_time
                })

    # 保存信息
    info_path = os.path.join(checkpoint_dir, 'checkpoints_info.json')
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=4)

    print(f"Checkpoints info saved to: {info_path}")


# ==================== 新增：检查点加载函数 ====================
def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, device='cpu'):
    """
    加载检查点恢复训练

    Args:
        checkpoint_path: 检查点路径
        model: 模型实例
        optimizer: 优化器实例
        scheduler: 学习率调度器实例
        device: 设备

    Returns:
        恢复的训练状态
    """
    print(f"Loading checkpoint from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 恢复模型权重
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # 如果只有模型权重
        model.load_state_dict(checkpoint)

    model.to(device)

    # 恢复优化器状态
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # 恢复调度器状态
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # 返回其他训练信息
    restored_state = {
        'epoch': checkpoint.get('epoch', 0),
        'best_acc': checkpoint.get('best_acc', 0.0),
        'train_losses': checkpoint.get('train_losses', []),
        'val_accuracies': checkpoint.get('val_accuracies', []),
        'class_indices': checkpoint.get('class_indices', {}),
        'config': checkpoint.get('config', {}),
        'is_finetune_phase': checkpoint.get('is_finetune_phase', False),
    }

    print(f"Checkpoint loaded. Epoch: {restored_state['epoch']}, "
          f"Best accuracy: {restored_state['best_acc']:.4f}")

    return restored_state


if __name__ == '__main__':
    main()