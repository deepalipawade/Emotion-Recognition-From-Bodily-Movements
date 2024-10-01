# feature_extraction.py
import os


import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2
from pytorchvideo.models.hub import i3d_r50

# Ensure torch home directory is set
os.environ['TORCH_HOME'] = '/home/mler24_team001/cluster-tutorial/cache'

class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class I3DModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = i3d_r50(pretrained=True)
        self.model.blocks[5].proj = Identity()  

    def forward(self, x):
        x = self.model(x)
        return x

class VideoDataset(Dataset):
    def __init__(self, video_paths, labels, transform=None, max_frames=32):
        self.video_paths = video_paths
        self.labels = labels
        self.transform = transform
        self.max_frames = max_frames

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        frames = []
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()

        if len(frames) > self.max_frames:
            frames = frames[::len(frames)//self.max_frames][:self.max_frames]
        else:
            frames += [frames[-1]] * (self.max_frames - len(frames))

        if self.transform:
            frames = [self.transform(frame) for frame in frames]

        frames = torch.stack(frames)
        frames = frames.permute(1, 0, 2, 3)  

        return frames, label

def load_dataset(txt_file):
    video_paths = []
    labels = []
    with open(txt_file, 'r') as file:
        for line in file:
            path, label = line.strip().split()
            video_paths.append(path)
            labels.append(int(label))
    return video_paths, labels

def extract_embeddings(model, dataloader, device):
    model.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for videos, label in dataloader:
            videos = videos.to(device)
            embedding = model(videos)
            embedding = embedding.cpu().numpy()
            embeddings.append(embedding)
            labels.append(label.cpu().numpy())
    embeddings = np.vstack(embeddings)
    labels = np.hstack(labels)
    return embeddings, labels

# Load datasets
train_txt = '/home/mler24_team001/cluster-tutorial/DataSet/Strategy2/Approach4/train_power.txt'
test_txt = '/home/mler24_team001/cluster-tutorial/DataSet/Strategy2/Approach4/test_power.txt'
val_txt = '/home/mler24_team001/cluster-tutorial/DataSet/Strategy2/Approach4/val_power.txt'

train_video_paths, train_labels = load_dataset(train_txt)
test_video_paths, test_labels = load_dataset(test_txt)
val_video_paths, val_labels = load_dataset(val_txt)

# Define transforms
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
])

# Create datasets and dataloaders
train_dataset = VideoDataset(train_video_paths, train_labels, transform=transform)
test_dataset = VideoDataset(test_video_paths, test_labels, transform=transform)
val_dataset = VideoDataset(val_video_paths, val_labels, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)

# Initialize model and device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
i3d_model = I3DModel().to(device)

# Extract embeddings
train_i3d_embeddings, train_labels = extract_embeddings(i3d_model, train_loader, device)
np.save('/home/mler24_team001/cluster-tutorial/features/final_dataset/new_approach/train_i3d_embeddings_4_power.npy', train_i3d_embeddings)
np.save('/home/mler24_team001/cluster-tutorial/features/final_dataset/new_approach/train_labels_4_power.npy', train_labels)

test_i3d_embeddings, test_labels = extract_embeddings(i3d_model, test_loader, device)
np.save('/home/mler24_team001/cluster-tutorial/features/final_dataset/new_approach/test_i3d_embeddings_4_power.npy', test_i3d_embeddings)
np.save('/home/mler24_team001/cluster-tutorial/features/final_dataset/new_approach/test_labels_4_power.npy', test_labels)

val_i3d_embeddings, val_labels = extract_embeddings(i3d_model, val_loader, device)
np.save('/home/mler24_team001/cluster-tutorial/features/final_dataset/new_approach/val_i3d_embeddings_4_power.npy', val_i3d_embeddings)
np.save('/home/mler24_team001/cluster-tutorial/features/final_dataset/new_approach/val_labels_4_power.npy', val_labels) 
