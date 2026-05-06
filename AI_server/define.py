import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import csv
import cv2
from tqdm import tqdm
import albumentations as A
import random


ROOT = Path.cwd() / "data_daxuli"  # Default root directory for data; can be overridden in notebooks
TRAIN_MANIFEST = ROOT / "train" / "manifest.json"
VAL_MANIFEST = ROOT / "validation" / "manifest.json"
TEST_MANIFEST = ROOT / "test" / "manifest.json"


# VN-safe preset: keep Vietnamese marks, reduce aggressive deformation
VN_SAFE_CONFIG = {
    "aug": {
        "p_apply": 0.5,
        "shear_range": (-0.4, 0.4),
        "rotation_range": (-5.0, 5.0),
        "elastic_sigma": [3, 5],
        "elastic_alpha": [30, 50],
        "geometric_distort_limit": 0.2,
        "geometric_perspective_scale": (0.01, 0.03),
    },
    "tta": {
        "num_shear": 4,
        "num_rot": 4,
        "shear_range": (-0.2, 0.2),
        "rotation_range": (-1.0, 1.0),
        "lambda": 1.0,
        "omega": 0.25,
    },
    "label": {
        "lowercase": True,
        "remove_punctuation": False,
    },
}

def ensure_binary_black_bg(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Làm mờ nhẹ để nối các nét chữ bị đứt do biến dạng/resize
    img_blurred = cv2.GaussianBlur(img, (1, 1), 0)
    
    # 2. Sử dụng Otsu để nhị phân hóa tự động
    _, out = cv2.threshold(img_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3. KIỂM TRA POLARITY: Đảm bảo nền luôn đen (0), chữ luôn trắng (255)
    h, w = out.shape
    corners = [out[0, 0], out[0, -1], out[-1, 0], out[-1, -1]]
    if np.mean(corners) > 127:
        out = 255 - out
        
    return out


def shear_image(img: np.ndarray, k: float):
    h, w = img.shape[:2]
    # x' = x + k*y
    M = np.array([[1.0, k, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    new_w = int(w + abs(k) * h)
    if k < 0:
        M[0, 2] = abs(k) * h
    sheared = cv2.warpAffine(
        img,
        M,
        (new_w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    # Back to original width for stable training shape
    sheared = cv2.resize(sheared, (w, h), interpolation=cv2.INTER_LINEAR)
    return ensure_binary_black_bg(sheared)


def rotate_image(img: np.ndarray, angle_deg: float):
    if A is None: return ensure_binary_black_bg(img)
    h, w = img.shape[:2]
    
    # SafeRotate tự động tính toán để không mất chữ ở góc
    aug = A.Compose([
        A.SafeRotate(limit=(angle_deg, angle_deg), p=1.0, border_mode=0)
    ])
    
    rotated = aug(image=img)["image"]
    # Ép về kích thước chuẩn 64x2048
    rotated = cv2.resize(rotated, (w, h), interpolation=cv2.INTER_LINEAR)
    return ensure_binary_black_bg(rotated)


def elastic_image(img: np.ndarray, sigma: int, alpha: int):
    if A is None:
        return ensure_binary_black_bg(img)
    aug = A.ElasticTransform(alpha=float(alpha), sigma=float(sigma), p=1.0)
    out = aug(image=img)["image"]
    return ensure_binary_black_bg(out)


def geometric_image(img: np.ndarray):
    if A is None:
        return ensure_binary_black_bg(img)
    
    h, w = img.shape[:2]
    
    # Bước 1: Thêm Padding đen ở biên trái/phải/trên/dưới (ví dụ 30px) 
    # để bảo vệ các ký tự sát lề khỏi bị cắt khi biến dạng
    aug_pipeline = A.Compose([
        A.PadIfNeeded(min_height=h, min_width=w + 200, border_mode=0, position='center'),
        A.OneOf([
            A.OpticalDistortion(distort_limit=VN_SAFE_CONFIG["aug"]["geometric_distort_limit"], p=1.0),
            A.GridDistortion(num_steps=5, distort_limit=VN_SAFE_CONFIG["aug"]["geometric_distort_limit"], p=1.0),
            A.Perspective(scale=VN_SAFE_CONFIG["aug"]["geometric_perspective_scale"], p=1.0),
        ], p=1.0),
    ])
    
    out = aug_pipeline(image=img)["image"]
    
    # Bước 2: Resize về lại (w, h)
    out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)
    return ensure_binary_black_bg(out)


def apply_single_aug_paper(img: np.ndarray, p_apply: float = None):
    if p_apply is None:
        p_apply = VN_SAFE_CONFIG["aug"]["p_apply"]

    if random.random() > p_apply:
        return ensure_binary_black_bg(img), "none"

    op = random.choice(["shear", "rotation", "elastic", "geometric"])

    if op == "shear":
        lo, hi = VN_SAFE_CONFIG["aug"]["shear_range"]
        k = random.uniform(lo, hi)
        return shear_image(img, k), f"shear(k={k:.3f})"

    if op == "rotation":
        lo, hi = VN_SAFE_CONFIG["aug"]["rotation_range"]
        theta = random.uniform(lo, hi)
        return rotate_image(img, theta), f"rotation(theta={theta:.2f})"

    if op == "elastic":
        sigma = random.choice(VN_SAFE_CONFIG["aug"]["elastic_sigma"])
        alpha = random.choice(VN_SAFE_CONFIG["aug"]["elastic_alpha"])
        return elastic_image(img, sigma, alpha), f"elastic(sigma={sigma}, alpha={alpha})"

    return geometric_image(img), "geometric(distort/stretch/perspective)"


class HandwritingDataset(Dataset):

    def __init__(self, image_paths, labels, char_to_idx, is_training=True):

        self.image_paths = image_paths
        self.labels = labels
        self.char_to_idx = char_to_idx
        self.is_training = is_training
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load and preprocess image
        img_path = self.image_paths[idx]
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise FileNotFoundError(f"Cannot read image at index {idx}: {img_path}")
        
        if self.is_training:
            img, aug_name = apply_single_aug_paper(img, p_apply=0.5)
        


        # 4. Normalize và định dạng Tensor
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0) # (1, H, W)

 
        label = self.labels[idx]
        label_indices = [self.char_to_idx[c] for c in label if c in self.char_to_idx]
        
        return torch.FloatTensor(img), torch.LongTensor(label_indices), self.image_paths[idx]

    @staticmethod
    def collate_fn(batch):
        images, labels, paths = zip(*batch)
        images        = torch.stack(images, dim=0)  # (B, 1, 64, 2048)
        label_lengths = torch.LongTensor([len(lbl) for lbl in labels])
        labels_concat = torch.cat(labels)
        return images, labels_concat, label_lengths, list(paths)

    @staticmethod
    def load_labels_csv(csv_path):
        image_paths = []
        labels = []

        # Use utf-8-sig so BOM-prefixed headers (\ufeffimage_path) are parsed correctly.
        with open(csv_path, mode='r', encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_path = (row.get('image_path') or row.get('\ufeffimage_path') or '').strip()
                text = row.get('text') or row.get('label') or ''
                exists_val = (row.get('exists') or '').strip().lower()

                if not image_path:
                    continue
                if exists_val and exists_val not in {'true', '1', 'yes'}:
                    continue
                if not Path(image_path).exists():
                    continue

                image_paths.append(image_path)
                labels.append(text)

        return image_paths, labels

class CNNBiLSTMCTC(nn.Module):
    def __init__(self, num_classes, input_height=100):
        super(CNNBiLSTMCTC, self).__init__()

        self.num_classes = num_classes

        # ==================== CNN BLOCK ====================
        # Block 1: 32 filters — học các cạnh cơ bản
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1   = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn2   = nn.BatchNorm2d(32)

        self.pool1     = nn.MaxPool2d(kernel_size=2, stride=2)  # H/2, W/2
        self.dropout1  = nn.Dropout(0.2)

        # Block 2: 64 filters — học móc, góc, nét cong
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3   = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4   = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn5   = nn.BatchNorm2d(64)

        self.conv6 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn6   = nn.BatchNorm2d(64)

        self.pool2    = nn.MaxPool2d(kernel_size=2, stride=2)   # H/4, W/4
        self.dropout2 = nn.Dropout(0.3)

        # Block 3: 128 filters — học đặc trưng phức tạp (dấu thanh, nét phụ)
        # Chia thành 3 cặp conv, mỗi cặp có residual connection
        # để tránh vanishing gradient qua 6 conv liên tiếp
        self.conv7  = nn.Conv2d(64,  128, kernel_size=3, stride=1, padding=1)
        self.bn7    = nn.BatchNorm2d(128)
        self.conv8  = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn8    = nn.BatchNorm2d(128)

        self.conv9  = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn9    = nn.BatchNorm2d(128)
        self.conv10 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn10   = nn.BatchNorm2d(128)

        self.conv11 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn11   = nn.BatchNorm2d(128)
        self.conv12 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn12   = nn.BatchNorm2d(128)

        # Projection 64→128 cho residual đầu tiên (block 2 out → block 3 in)
        # Vì conv7 thay đổi channels từ 64 → 128, cần project để cộng được
        self.res_proj = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0)

        # FIX 2: Bật lại dropout sau block 3
        self.dropout3 = nn.Dropout(0.3)

        # After column-wise pooling: mỗi time-step có 128 features
        # (max_pool + avg_pool cộng lại, không concat → giữ nguyên 128)
        self.rnn_input_size = 128

        # ==================== BiLSTM BLOCK ====================
        self.lstm1 = nn.LSTM(
            input_size=self.rnn_input_size,
            hidden_size=256,
            num_layers=1,
            bidirectional=True,
            batch_first=False
        )

        # FIX 3: Dropout giữa 2 LSTM
        self.dropout_lstm = nn.Dropout(0.3)

        self.lstm2 = nn.LSTM(
            input_size=512,   # 256 * 2 (bidirectional)
            hidden_size=256,
            num_layers=1,
            bidirectional=True,
            batch_first=False
        )

        # ==================== CTC OUTPUT ====================
        self.fc1         = nn.Linear(512, 512)
        self.dropout_fc  = nn.Dropout(0.5)
        self.fc2         = nn.Linear(512, num_classes)

    def forward(self, x):
        # ── Block 1 ──────────────────────────────────────────
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        # shape: (B, 32, H/2, W/2)

        # ── Block 2 ──────────────────────────────────────────
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        # shape: (B, 64, H/4, W/4)

        # ── Block 3  ─────────────────
        residual = self.res_proj(x)            # (B, 128, H/4, W/4)
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = x + residual                       # skip connection #1

        residual = x
        x = F.relu(self.bn9(self.conv9(x)))
        x = F.relu(self.bn10(self.conv10(x)))
        x = x + residual                       # skip connection #2

        residual = x
        x = F.relu(self.bn11(self.conv11(x)))
        x = F.relu(self.bn12(self.conv12(x)))
        x = x + residual                       # skip connection #3

        # FIX 2: Dropout sau block 3 (trước đây bị comment)
        x = self.dropout3(x)
        # shape: (B, 128, H/4, W/4)

        # ── Column-wise pooling ───────────────────────────────
        # FIX 4: Kết hợp max + avg pool thay vì chỉ dùng max
        # Max pool: bắt đặc trưng nổi bật nhất theo chiều H
        # Avg pool: giữ phân bố tổng thể — quan trọng cho dấu thanh nhỏ
        # Cộng lại (không concat) để giữ rnn_input_size = 128
        x_max = F.max_pool2d(x, kernel_size=(x.size(2), 1))  # (B, 128, 1, W/4)
        x_avg = F.avg_pool2d(x, kernel_size=(x.size(2), 1))  # (B, 128, 1, W/4)
        x = x_max + x_avg                                     # (B, 128, 1, W/4)

        # ── Reshape cho RNN ───────────────────────────────────
        x = x.squeeze(2)        # (B, 128, W/4)
        x = x.permute(2, 0, 1)  # (W/4, B, 128) — time-first cho LSTM

        # ── BiLSTM ────────────────────────────────────────────
        x, _ = self.lstm1(x)    # (T, B, 512)

        # FIX 3: Dropout giữa 2 LSTM (trước đây bị comment)
        x = self.dropout_lstm(x)

        x, _ = self.lstm2(x)    # (T, B, 512)

        # ── Output ────────────────────────────────────────────
        x = self.fc1(x)         # (T, B, 512)

        # FIX 5: Bỏ ReLU trước fc2
        # ReLU clip giá trị âm → mất thông tin trước log_softmax
        x = self.dropout_fc(x)

        x = self.fc2(x)         # (T, B, num_classes)

        # Log softmax tính ngoài (trong CTCLoss hoặc decoder)
        return x

