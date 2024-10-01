import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torchvision.transforms as transforms
import cv2
from PIL import Image
import numpy as np


# Here, Consider the SlowFast ResNet50 model, operating on (8×256×256, 32×256×256) video clips, 
# respectively for the slow and fast patway, and giving rises to embedding vectors of dimension 2304.

os.environ['TORCH_HOME'] = '/home/mler24_team001/cluster-tutorial/cache'

class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class SlowFastModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.hub.load("facebookresearch/pytorchvideo", "slowfast_r50", pretrained=True)
        self.model.blocks[6].proj = Identity()

    def forward(self, x):
        x = self.model(x)
        return x

class PackPathway(nn.Module):
    """
    Transform for converting video frames as a list of tensors for SlowFast model.
    """
    def __init__(self, alpha_slowfast):
        super().__init__()
        self.alpha_slowfast = alpha_slowfast

    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // self.alpha_slowfast
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list

class VideoDataset(Dataset):
    def __init__(self, video_paths, labels, transform=None, max_frames=32, alpha_slowfast=4):
        self.video_paths = video_paths
        self.labels = labels
        self.transform = transform
        self.max_frames = max_frames
        self.alpha_slowfast = alpha_slowfast

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        # Load video frames using OpenCV
        frames = []
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()

        # Sample frames to max_frames
        if len(frames) > self.max_frames:
            frames = frames[::len(frames)//self.max_frames][:self.max_frames]
        else:
            frames += [frames[-1]] * (self.max_frames - len(frames))

        # Apply transformations
        if self.transform:
            frames = [self.transform(frame) for frame in frames]

        frames = torch.stack(frames)
        frames = frames.permute(1, 0, 2, 3)  # Change from (T, C, H, W) to (C, T, H, W)
        
        # Pack frames for SlowFast model
        frames = PackPathway(self.alpha_slowfast)(frames)

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
            videos = [v.to(device) for v in videos]
            embedding = model(videos)
            embedding = embedding.cpu().numpy()
            embeddings.append(embedding)
            labels.append(label.cpu().numpy())
    embeddings = np.vstack(embeddings)
    labels = np.hstack(labels)
    return embeddings, labels

# Load datasets
train_txt = '/home/mler24_team001/cluster-tutorial/DataSet/train.txt'
test_txt = '/home/mler24_team001/cluster-tutorial/DataSet/test.txt'
val_txt = '/home/mler24_team001/cluster-tutorial/DataSet/val.txt'

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
slowfast_model = SlowFastModel().to(device)

# Extract embeddings
train_slowfast_embeddings, train_labels = extract_embeddings(slowfast_model, train_loader, device)
np.save('/home/mler24_team001/cluster-tutorial/features/dataset/train_slowfast_embeddings.npy', train_slowfast_embeddings)
np.save('/home/mler24_team001/cluster-tutorial/features/dataset/train_labels.npy', train_labels)

test_slowfast_embeddings, test_labels = extract_embeddings(slowfast_model, test_loader, device)
np.save('/home/mler24_team001/cluster-tutorial/features/dataset/test_slowfast_embeddings.npy', test_slowfast_embeddings)
np.save('/home/mler24_team001/cluster-tutorial/features/dataset/test_labels.npy', test_labels)

val_slowfast_embeddings, val_labels = extract_embeddings(slowfast_model, val_loader, device)
np.save('/home/mler24_team001/cluster-tutorial/features/dataset/val_slowfast_embeddings.npy', test_slowfast_embeddings)
np.save('/home/mler24_team001/cluster-tutorial/features/dataset/val_labels.npy', test_labels)