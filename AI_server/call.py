import torch
from pathlib import Path
from torch.utils.data import DataLoader
import random
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_dir = Path.cwd()
checkpoint_dir = base_dir / "checkpoints"

best_model_path = checkpoint_dir / "best_model.pth"
last_model_path = checkpoint_dir / "last_model.pth"

if best_model_path.exists():
    checkpoint_path = best_model_path
elif last_model_path.exists():
    checkpoint_path = last_model_path
else:
    raise FileNotFoundError(
        f"No checkpoint found. Expected one of: {best_model_path} or {last_model_path}"
    )

ckpt = torch.load(str(checkpoint_path), map_location=device)

if isinstance(ckpt, dict) and "char_to_idx" in ckpt:
    char_to_idx = ckpt["char_to_idx"]
else:
    raise KeyError(
        "Checkpoint does not contain 'char_to_idx'. Re-train or save vocabulary in checkpoint."
    )

idx_to_char = {idx: char for char, idx in char_to_idx.items()}
num_classes = len(char_to_idx)

model = CNNBiLSTMCTC(num_classes=num_classes).to(device)

if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
else:
    # Backward compatibility: checkpoint might be a raw state_dict
    model.load_state_dict(ckpt)

model.eval()
print(f"Loaded checkpoint: {checkpoint_path}")

data_root = base_dir / "data_daxuli"
test_csv = data_root / "test_labels.csv"

if not test_csv.exists():
    raise FileNotFoundError(f"Missing test CSV: {test_csv}")

test_image_paths, test_labels = load_labels_csv(test_csv)
if not test_image_paths:
    raise RuntimeError("No valid test samples loaded from CSV.")

test_dataset = HandwritingDataset(
    image_paths=test_image_paths,
    labels=test_labels,
    char_to_idx=char_to_idx,
    is_training=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=0,
    pin_memory=(device.type == "cuda"),
    collate_fn=collate_fn
)

all_pred_texts = []
all_target_texts = []
all_image_paths = []

with torch.no_grad():
    for images, labels, label_lengths, paths in test_loader: # <--- Nhận thêm paths
        images = images.to(device, non_blocking=True)
        logits = model(images)
        
        # log_softmax và decode như cũ
        log_probs = F.log_softmax(logits, dim=2)
        pred_texts = beam_search_decode(log_probs, idx_to_char, blank_idx=0, beam_width=10)
        target_texts = targets_to_strings(labels, label_lengths, idx_to_char)

        all_pred_texts.extend(pred_texts)
        all_target_texts.extend(target_texts)
        all_image_paths.extend(paths) 

metrics = compute_ocr_metrics(all_pred_texts, all_target_texts)
print(f"Test samples: {len(all_target_texts):,}")
print(f"Sequence Accuracy: {metrics['seq_acc']:.2%}")
print(f"CER: {metrics['cer']:.2%}")
print(f"WER: {metrics['wer']:.2%}")
print(f"Normalized Edit Similarity: {metrics['norm_edit_similarity']:.2%}")


print("\nSample predictions:")
indices = random.sample(range(len(all_pred_texts)), min(20, len(all_pred_texts)))

for i, idx in enumerate(indices, start=1):
    pred   = all_pred_texts[idx]
    target = all_target_texts[idx]
    img_path = str(all_image_paths[idx])
    match  = "✅" if pred == target else "❌"

    # Đọc ảnh
    img = cv2.imread(img_path)
    if img is None:
        print(f"Can't read image: {img_path}")
        continue
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # In thông tin văn bản
    print(f"{i:02d}. {match} PRED: {pred}")
    print(f"    GT  : {target}")

    # Hiển thị ảnh ngay lập tức
    plt.figure(figsize=(10, 2)) # Chỉnh kích thước khung hình (rộng, cao)
    plt.imshow(img_rgb)
    plt.axis('off') # Ẩn trục tọa độ x, y
    plt.show() 
    print("-" * 50)