# TAOKD

TAOKD (Trajectory-Aware Online Knowledge Distillation) is a novel framework that captures and aligns the dynamic evolution of key features across layers using a deformable attention mechanism for precise intermediate-level knowledge transfer.

## Features
- **Multi-Dataset Support**: CIFAR-100, Tiny-ImageNet, Stanford Cars, NABirds
- **Rich Model Architectures**: ResNet, MobileNetV2, Wide ResNet

## Project Structure

```
TAOKD/
├── Train.py                           # Main training script
├── run.py                             # Hyperparameter search and experiment management
├── compute_loss.py                    # Loss function implementations
├── deformable_attention.py            # Deformable attention module
├── dataset_process.py                 # Dataset processing
├── model/                             # Model definitions directory
│   ├── resnet_cifar.py               # ResNet models for CIFAR
│   ├── resnet_tiny_imagenet_resnet34.py
│   ├── resnet_stanford_cars_nabirds.py
│   ├── mobilenetv2_cifar.py
│   ├── mobilenetv2_tiny_imagenet.py
│   └── wrn_cifar.py                  # Wide ResNet models
└── checkpoint/                        # Training checkpoints and results output
```

## Requirements

```bash
pip install torch torchvision transformers tqdm einops timm
```

## Quick Start

### Single Training Run

```bash
python Train.py \
    --dataset_type tiny-imagenet \
    --distill 1 \
    --epochs 200 \
    --lr 0.01 \
    --temperature 1 \
    --momentum 0.9 \
    --weight_decay 1e-4 \
    --scheduler_milestones 100 140 180 \
    --gamma 0.1 \
    --n_heads 2 \
    --n_groups 2 \
    --stride 1 \
    --alpha 0.1 \
    --beta 0.1 \
    --use_aligned_loss True \
    --use_evo_loss True \
    --output_dir ./checkpoint/exp1
```

### Hyperparameter Search

Use `run.py` for batch experiments and hyperparameter search:

```bash
python run.py
```

Modify the `param_grid` dictionary in `run.py` to define your search space.

## Supported Datasets

| Dataset | Classes | Image Size | Parameter `--dataset_type` |
|---------|---------|------------|---------------------------|
| CIFAR-100 | 100 | 32x32 | `cifar` |
| Tiny-ImageNet | 200 | 64x64 | `tiny-imagenet` |
| Stanford Cars | 196 | 224x224 | `stanford-cars` |
| NABirds | 555 | 224x224 | `nabirds` |

## Training Parameters

### Basic Parameters

| Parameter | Description | Default         |
|-----------|-------------|-----------------|
| `--dataset_type` | Dataset type | `tiny-imagenet` |
| `--distill` | Enable knowledge distillation | `1`             |
| `--epochs` | Number of training epochs | `200`           |
| `--batch_size` | Batch size | `32`            |
| `--lr` | Learning rate | `0.01`          |

### Knowledge Distillation Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--temperature` | Distillation temperature | `1` |
| `--alpha` | Aligned Loss weight | `0.1` |
| `--beta` | Evolution Loss weight | `0.1` |
| `--use_aligned_loss` | Use Aligned Loss | `True` |
| `--use_evo_loss` | Use Evolution Loss | `True` |

### Deformable Attention Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--n_heads` | Number of attention heads | `2` |
| `--n_groups` | Number of attention groups | `2` |
| `--stride` | Sampling stride | `1` |

## Loss Function

The total loss function for the student model consists of the following components:

```
L_total = (1 - lambda) * L_ce + lambda * L_kd + alpha * L_aligned + beta * L_evo
```

- **L_ce**: Cross-entropy loss
- **L_kd**: KL-divergence knowledge distillation loss
- **L_aligned**: Feature alignment loss (via deformable attention)
- **L_evo**: Attention evolution consistency loss

The teacher model is simultaneously optimized using cross-entropy loss and KL-divergence loss.

## Model Architectures

The project supports the following model architectures:

- **ResNet**: resnet8x4, resnet32x4 (CIFAR), resnet18, resnet34, resnet50
- **MobileNetV2**: With different width multipliers and temperature parameters
- **Wide ResNet**: wrn_16_2, wrn_40_2

Modify `student_model` and `teacher_model` selection in `Train.py`.

## Output Files

The training process generates the following in the specified `output_dir`:

- `student_model.pth`: Best checkpoint for student model
- `teacher_model.pth`: Best checkpoint for teacher model
- Training logs: Loss and accuracy information for each epoch

## Hyperparameter Search Results

After running batch experiments with `run.py`, results are saved in `checkpoint/results.csv`:

| Column | Description |
|--------|-------------|
| Run_ID | Unique experiment identifier |
| Params | Hyperparameter configuration |
| Accuracy | Final student model accuracy |
| Loss | Final student model loss |
| Log_Path | Path to log file |

## Citation

If you use this project in your research, please consider citing the relevant papers.
