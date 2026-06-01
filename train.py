import os
import torch
import random
import numpy as np
import transformers
from tqdm import tqdm
import torch.nn as nn
from typing import Optional, List
import torch.optim as optim
from dataclasses import dataclass, field
from dataset_process import make_supervised_data_module
from compute_loss import student_distillationloss, teacher_distillationloss
from model.resnet_cifar import resnet8x4, resnet32x4
from model.resnet_tiny_imagenet_resnet34 import tiny_imagenet_resnet34
from model.mobilenetv2_cifar import mobile_half
from model.mobilenetv2_tiny_imagenet import mobilenetv2_T_w
from model.wrn_cifar import wrn_16_2, wrn_40_2
from model.resnet_stanford_cars_nabirds import resnet18, resnet50

local_rank = None

def seed(seed=5):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

@dataclass
class DataArguments:
    dataset_type: str = field(default="stanford-cars", metadata={"help": "Select dataset type"})
    train_dir: Optional[str] = field(default=None)
    test_dir: Optional[str] = field(default=None)
    batch_size: int = field(default=16)
    num_workers: int = field(default=4)
    use_coarse: bool = field(default=False)

    def __post_init__(self):
        base_path = "/root/TAOKD/data"
        if self.dataset_type.lower() == "stanford-cars":
            self.train_dir = self.train_dir or os.path.join(base_path, "stanford-cars/train")
            self.test_dir = self.test_dir or os.path.join(base_path, "stanford-cars/test")
        elif self.dataset_type.lower() == "cifar":
            self.train_dir = self.train_dir or os.path.join(base_path, "cifar100/train")
            self.test_dir = self.test_dir or os.path.join(base_path, "cifar100/test")
        elif self.dataset_type.lower() == "tiny-imagenet":
            self.train_dir = self.train_dir or os.path.join(base_path, "tiny-imagenet/train")
            self.test_dir = self.test_dir or os.path.join(base_path, "tiny-imagenet/test")
        elif self.dataset_type.lower() == "nabirds":
            self.train_dir = self.train_dir or os.path.join(base_path, "nabirds/train")
            self.test_dir = self.test_dir or os.path.join(base_path, "nabirds/test")
        else:
            raise ValueError(f"Unsupported dataset type: {self.dataset_type}")

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    distill: int = field(default=None)
    epochs: int = field(default=None)
    temperature: float = field(default=None)
    lr: float = field(default=None)
    momentum: float = field(default=None)
    weight_decay: float = field(default=None)
    scheduler_milestones: Optional[List[int]] = field(default=None)
    gamma: float = field(default=None)
    n_heads: int = field(default=None)
    n_groups: int = field(default=None)
    stride: int = field(default=None)
    alpha: float = field(default=None)
    beta: float = field(default=None)
    use_aligned_loss: bool = field(default=True)
    use_evo_loss: bool = field(default=True)
    output_dir: str = field(default=None)

def evaluate_model(model, dataloader, device, label_key, model_name="Model"):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating {model_name}", unit="batch"):
            images = batch["images"].to(device)
            labels = batch[label_key].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    return acc

def train():
    seed()
    global local_rank
    parser = transformers.HfArgumentParser((DataArguments, TrainingArguments))
    data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n===== Data Arguments =====")
    for field_name in data_args.__dataclass_fields__:
        print(f"{field_name}: {getattr(data_args, field_name)}")

    print("\n===== Training Arguments =====")
    custom_fields = [
        "distill", "epochs", "temperature", "lr", "momentum",
        "weight_decay", "scheduler_milestones", "gamma",
        "n_heads", "n_groups", "stride", "alpha", "beta", "use_aligned_loss", "use_evo_loss", "output_dir"
    ]
    for field_name in custom_fields:
        print(f"{field_name}: {getattr(training_args, field_name)}")

    print(f"Current dataset type: {data_args.dataset_type}")
    if data_args.dataset_type.lower() == "stanford-cars":
        num_classes = 196
        label_key = "labels"
        image_size = 224
    elif data_args.dataset_type.lower() == "tiny-imagenet":
        num_classes = 200
        label_key = "labels"
        image_size = 64
    elif data_args.dataset_type.lower() == "nabirds":
        num_classes = 555
        label_key = "labels"
        image_size = 224
    elif data_args.dataset_type.lower() == "cifar":
        num_classes = 100 if not data_args.use_coarse else 20
        label_key = "coarse_idx" if data_args.use_coarse else "fine_idx"
        image_size = 32
    else:
        raise ValueError(f"Unsupported dataset type: {data_args.dataset_type}")

    print(f"num_classes={num_classes}, image_size={image_size}")

    data_module = make_supervised_data_module(data_args=data_args)
    train_loader = data_module["train_dataloader"]
    test_loader = data_module["test_dataloader"]

    student_model_name = "resnet18"
    student_model = resnet18(num_classes=num_classes)
    # student_model = mobilenetv2_T_w(T=6, W=1.4, feature_dim=num_classes)
    print(f"Student model: {student_model_name}")

    if training_args.distill == 1:
        teacher_model_name = "resnet50"
        teacher_model = resnet50(num_classes=num_classes)
        # teacher_model = mobilenetv2_T_w(T=6, W=1.4, feature_dim=num_classes)
        print(f"Teacher model: {teacher_model_name}")

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel!")
        student_model = nn.DataParallel(student_model)
        teacher_model = nn.DataParallel(teacher_model)

    student_model = student_model.to(device)
    teacher_model = teacher_model.to(device)

    student_criterion = student_distillationloss(training_args, device).to(device)
    teacher_criterion = teacher_distillationloss(training_args, device).to(device)

    optimizer_student = optim.SGD(
        list(student_model.parameters()) + list(student_criterion.parameters()),
        lr=training_args.lr,
        momentum=training_args.momentum,
        weight_decay=training_args.weight_decay
    )

    scheduler_student = optim.lr_scheduler.MultiStepLR(optimizer_student,
                                               milestones=training_args.scheduler_milestones,
                                               gamma=training_args.gamma)
    optimizer_teacher = optim.SGD(
        list(teacher_model.parameters()),
        lr=training_args.lr,
        momentum=training_args.momentum,
        weight_decay=training_args.weight_decay
    )
    scheduler_teacher = optim.lr_scheduler.MultiStepLR(optimizer_teacher,
                                                       milestones=training_args.scheduler_milestones,
                                                       gamma=training_args.gamma)

    best_acc = 0.0
    os.makedirs(training_args.output_dir, exist_ok=True)
    student_save_path = os.path.join(training_args.output_dir, "student_model.pth")
    teacher_save_path = os.path.join(training_args.output_dir, "teacher_model.pth")

    for epoch in range(training_args.epochs):
        print(f"\n===== Epoch {epoch + 1}/{training_args.epochs} =====")
        student_model.train()
        teacher_model.train()

        running_loss_student = 0.0
        running_loss_teacher = 0.0

        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}", unit="batch"):
            images = batch["images"].to(device)
            labels = batch[label_key].to(device)

            optimizer_student.zero_grad()
            optimizer_teacher.zero_grad()

            student_feats, student_logits = student_model(images, is_feat=True, preact=True)
            teacher_feats, teacher_logits = teacher_model(images, is_feat=True, preact=True)

            student_loss = student_criterion(student_feats, student_logits, teacher_feats, teacher_logits, labels)
            running_loss_student += student_loss.item()

            teacher_loss = teacher_criterion(teacher_logits, student_logits, labels)
            running_loss_teacher += teacher_loss.item()

            total_loss = student_loss + teacher_loss
            total_loss.backward()

            optimizer_student.step()
            optimizer_teacher.step()

        scheduler_student.step()
        scheduler_teacher.step()

        avg_loss_student = running_loss_student / len(train_loader)
        avg_loss_teacher = running_loss_teacher / len(train_loader)

        print(f"Epoch {epoch + 1} | Student Loss: {avg_loss_student:.4f} | Student LR: {scheduler_student.get_last_lr()[0]:.6f}")
        print(f"Epoch {epoch + 1} | Teacher Loss: {avg_loss_teacher:.4f} | Teacher LR: {scheduler_teacher.get_last_lr()[0]:.6f}")
        # if (epoch + 1) % 10 == 0:

        print(f"\n--- Running Validation (Epoch {epoch + 1}) ---")
        current_student_acc = evaluate_model(
            student_model, test_loader, device, label_key, f"Student Model (Val. Epoch {epoch + 1})"
        )
        current_teacher_acc = evaluate_model(
            teacher_model, test_loader, device, label_key, f"Teacher Model (Val. Epoch {epoch + 1})"
        )
        print(f"Current Student Acc: {current_student_acc:.2f}%, Teacher Acc: {current_teacher_acc:.2f}%")

        if current_student_acc > best_acc:
            best_acc = current_student_acc
            print(f"New Best! Student: {best_acc:.2f}% (Teacher: {current_teacher_acc:.2f}%). Saving models...")
            torch.save(student_model.state_dict(), student_save_path)
            torch.save(teacher_model.state_dict(), teacher_save_path)


    print("\n" + "=" * 40)
    print("\nStarting testing...")
    student_model.load_state_dict(torch.load(student_save_path, weights_only=True))
    teacher_model.load_state_dict(torch.load(teacher_save_path, weights_only=True))

    final_s_acc = evaluate_model(student_model, test_loader, device, label_key, "Final Student")
    final_t_acc = evaluate_model(teacher_model, test_loader, device, label_key, "Final Teacher")

    print("\n===== FINAL EXPERIMENT RESULTS =====")
    print(f"Student Accuracy: {final_s_acc:.2f}%")
    print(f"Teacher Accuracy: {final_t_acc:.2f}%")

if __name__ == "__main__":
    train()
