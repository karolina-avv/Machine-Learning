import torch
import os
import librosa
import numpy as np
import torchaudio
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ThreadPoolExecutor
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18 

class AudioDataset(Dataset):
    def __init__(self, directory):
        """
        Args:
            directory (string): Path to the directory with all the audio files.
        """

        self.load_audio(directory)
        self.load_labels()

        # Use ThreadPoolExecutor to load and process audio in parallel
        max_workers = os.cpu_count()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(
                tqdm(
                    executor.map(self.load_and_process_audio, self.file_paths),
                    total=len(self.file_paths),
                    desc="Loading and processing audio files",
                )
            )

        # Unpack results
        self.melspectrograms = list(results)

    def load_audio(self, directory):
        self.file_paths = [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if f.endswith(".mp3")
        ]
        self.file_paths.sort(
            key=lambda f: int(os.path.splitext(os.path.basename(f))[0])
        )
        return self.file_paths

    def load_labels(self):
        self.labels = np.load("train_labels.npy", allow_pickle=True)
        return self.labels

    def load_and_process_audio(self, file_path):
        audio, sample_rate = librosa.load(file_path, sr=None, mono=True)
        audio_tensor = (
            torch.from_numpy(audio).float().unsqueeze(0)
        )  
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=2048,
            win_length=1024,
            hop_length=512,
            n_mels=128,
        )
        melspectrogram = mel_transform(audio_tensor)
        melspectrogram_db = torchaudio.transforms.AmplitudeToDB()(melspectrogram)
        return melspectrogram_db

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        melspectrogram_db = self.melspectrograms[idx]
        label = self.labels[idx]
        return melspectrogram_db, label
    
    def create_dataloader(dataset, batch_size=256, shuffle=True, split_ratio=0.8):

        if split_ratio == 1:
            # Create dataloader for the entire dataset
            dataloader = DataLoader(
                dataset, batch_size=batch_size, shuffle=shuffle
            )
            return dataloader, None
        
        # Split the dataset into train and test sets
        train_size = int(split_ratio * len(dataset))
        
        test_size = len(dataset) - train_size
        
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, test_size]
        )
        # Create dataloaders for train and test sets
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=shuffle
        )
        test_dataloader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )
        return train_dataloader, test_dataloader

class AudioClassifier(nn.Module):
    def __init__(self, num_classes):
        super(AudioClassifier, self).__init__()
        self.resnet = resnet18(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(512, num_classes) 

    def forward(self, x):
        return self.resnet(x)
    
    from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR


def do_nothing(x):
    return x


def train(train_loader, val_loader, lr=.0001,  num_epochs=20, transform=do_nothing):
    model = AudioClassifier(num_classes=4).to("cuda:0")
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=0.0001)  #trying .01 lr and .0005 decay
    scheduler = CosineAnnealingLR(optimizer, T_max=5)

    
    for epoch in range(num_epochs):  # Configurable number of epochs
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images = transform(images)
            images, labels = images.to("cuda:0"), labels.to("cuda:0")
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        

    return model

label_dataset=  "/Users/karolina/Desktop/Machine Learning/competition/train_label.txt"

with open(label_dataset, "r") as file:
    labels = file.readlines()

# Strip newline characters from labels
labels = [label.strip() for label in labels]

# Convert labels to numpy array
labels_array = np.array(labels)

# Save the numpy array to train_labels.npy
np.save("train_labels.npy", labels_array)

print("Labels converted and saved to train_labels.npy")

import pickle

#train_directory = "./train_mp3s"
train_directory = "/Users/karolina/Desktop/Machine Learning/competition/train_mp3s"
dataset = AudioDataset(train_directory)

# Define the filename for the binary file
filename = "train_dataset.bin"
with open(filename, "wb") as file:
    pickle.dump(dataset, file)
with open(filename, "rb") as file:
    dataset = pickle.load(file)