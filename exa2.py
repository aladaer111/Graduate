import os
import torch
import torch.nn as nn
import random
import torchaudio
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import accuracy_score, recall_score, f1_score

config = {
    "data_root": r"D:\Graduate Design\DP\DEEPSHIP222\DeepShip-main",  # 主目录路径
    "classes": ["Cargo", "Passenger", "Tanker", "Tug"],  # 子目录名称列表
    "sample_rate": 22050,  # 音频采样率
    "duration": 5,  # 截取音频时长（秒）
    "n_mels": 128,  # 梅尔频谱维度
    "batch_size": 16,
    "n_epochs": 50,  # 增加训练轮数
    "lr": 1e-4,
    "n_heads": 8,  # Transformer头数
    "num_layers": 4,  # Transformer层数
    "embed_dim": 256,
    "num_samples": 20,  # 每个类别使用样本数（小样本设置）
    "n_way": 4,  # N-way K-shot中的N
    "k_shot": 10  # N-way K-shot中的K
}

# 音频预处理
class AudioPreprocessor:
    def __init__(self, config):
        self.sr = config["sample_rate"]
        self.duration = config["duration"]
        self.n_mels = config["n_mels"]

    def __call__(self, file_path):
        # 加载音频
        waveform, sr = librosa.load(file_path, sr=self.sr)

        # 数据增强
        if random.random() < 0.5:
            noise = np.random.normal(0, 0.01, waveform.shape)
            waveform = waveform + noise

        if random.random() < 0.5:
            rate = random.uniform(0.8, 1.2)
            waveform = librosa.effects.time_stretch(waveform, rate=rate)

        if random.random() < 0.5:
            n_steps = random.randint(-2, 2)
            waveform = librosa.effects.pitch_shift(waveform, sr=sr, n_steps=n_steps)

        # 统一音频长度
        max_len = int(self.duration * self.sr)
        if len(waveform) > max_len:
            waveform = waveform[:max_len]
        else:
            waveform = np.pad(waveform, (0, max(0, max_len - len(waveform))))

        # 转换为梅尔频谱图
        mel_spec = librosa.feature.melspectrogram(
            y=waveform, sr=sr, n_mels=self.n_mels)
        log_mel = librosa.power_to_db(mel_spec)

        # 标准化
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-9)
        return torch.FloatTensor(log_mel)

# 数据集
class AudioDataset(Dataset):
    def __init__(self, config, mode="train"):
        self.config = config
        self.preprocessor = AudioPreprocessor(config)
        self.file_paths = []
        self.labels = []

        # 遍历每个类别目录
        for label_idx, class_name in enumerate(config["classes"]):
            class_dir = os.path.join(config["data_root"], class_name)
            files = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.endswith(".wav")]

            # 每个类取前num_samples个样本
            files = files[:config["num_samples"]] if mode == "train" else files[config["num_samples"]:]

            self.file_paths.extend(files)
            self.labels.extend([label_idx] * len(files))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        spec = self.preprocessor(self.file_paths[idx])
        return spec.unsqueeze(0), self.labels[idx]  # 添加通道维度

# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

# 基于Transformer的特征编码器
class AudioTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=config["embed_dim"],
            nhead=config["n_heads"]
        )
        self.transformer = nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=config["num_layers"]
        )
        self.projection = nn.Sequential(
            nn.Linear(config["n_mels"], config["embed_dim"]),
            nn.ReLU()
        )
        self.positional_encoding = PositionalEncoding(config["embed_dim"])
        self.classifier = nn.Linear(config["embed_dim"], len(config["classes"]))

    def forward(self, x):
        # x: [batch, 1, n_mels, time]
        x = x.squeeze(1).permute(0, 2, 1)  # [batch, time, n_mels]
        x = self.projection(x)  # [batch, time, embed_dim]
        x = self.positional_encoding(x)  # 添加位置编码
        x = x.permute(1, 0, 2)  # [time, batch, embed_dim]
        x = self.transformer(x)  # [time, batch, embed_dim]
        x = x.mean(dim=0)  # [batch, embed_dim]
        return self.classifier(x)

# 小样本原型网络改进（简化示例）
class PrototypeNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.feature_encoder = AudioTransformer(config)

    def forward(self, support_set, query_set):
        support_features = self.feature_encoder(support_set)
        query_features = self.feature_encoder(query_set)
        # 动态原型校准策略（简化）
        prototypes = support_features.mean(dim=0)
        distances = torch.cdist(query_features, prototypes)
        return -distances

# 训练准备
train_set = AudioDataset(config, mode="train")
test_set = AudioDataset(config, mode="test")
train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True)
test_loader = DataLoader(test_set, batch_size=config["batch_size"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AudioTransformer(config).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=1e-5)
scheduler = StepLR(optimizer, step_size=15, gamma=0.1)  # 调整学习率调度器参数

# 训练循环
for epoch in range(config["n_epochs"]):
    model.train()
    total_loss = 0
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # 验证
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    # 计算评价指标
    accuracy = accuracy_score(all_labels, all_preds)
    # 设置 zero_division 参数为 0
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro')

    # 更新学习率
    scheduler.step()

    print(f"Epoch [{epoch + 1}/{config['n_epochs']}] Loss: {total_loss / len(train_loader):.4f} Acc: {100 * accuracy:.2f}% Recall: {100 * recall:.2f}% F1: {100 * f1:.2f}%")