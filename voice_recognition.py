import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder

# ----------------------------
# CONFIG
# ----------------------------
dataset_path = r"D:/voice_dataset"   # Change this to your dataset path
sample_rate = 22050
n_mfcc = 13

# ----------------------------
# DATASET CLASS
# ----------------------------
class VoiceDataset(Dataset):
    def __init__(self, dataset_path):
        self.files = []
        self.labels = []
        self.speakers = []

        # Loop over each speaker directory
        for speaker in os.listdir(dataset_path):
            speaker_dir = os.path.join(dataset_path, speaker)
            if os.path.isdir(speaker_dir):
                wav_files = glob.glob(os.path.join(speaker_dir, "*.wav"))
                wav_files += glob.glob(os.path.join(speaker_dir, "*.WAV"))  # handle uppercase
                self.files.extend(wav_files)
                self.labels.extend([speaker] * len(wav_files))
                self.speakers.append(speaker)

        # Encode speaker names to numbers
        self.encoder = LabelEncoder()
        self.labels = self.encoder.fit_transform(self.labels)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        label = self.labels[idx]

        # Load audio
        signal, sr = librosa.load(file, sr=sample_rate)

        # Extract MFCCs
        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
        mfcc = np.mean(mfcc.T, axis=0)  # Average over time

        return torch.tensor(mfcc, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# ----------------------------
# SIMPLE MODEL
# ----------------------------
class VoiceClassifier(nn.Module):
    def __init__(self, num_classes):
        super(VoiceClassifier, self).__init__()
        self.fc1 = nn.Linear(n_mfcc, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# ----------------------------
# TRAINING LOOP
# ----------------------------
def train_model():
    # Load dataset
    dataset = VoiceDataset(dataset_path)

    if len(dataset) == 0:
        print("❌ No audio files found! Check dataset path and file extensions.")
        return

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Define model
    num_classes = len(set(dataset.labels))
    model = VoiceClassifier(num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training
    for epoch in range(10):  # Train for 10 epochs
        total_loss = 0
        for features, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

    print("✅ Training complete!")

    # Save model + class labels
    torch.save(model.state_dict(), "voice_model.pth")
    np.save("classes.npy", dataset.encoder.classes_)
    print("✅ Model saved as voice_model.pth and classes.npy")

# ----------------------------
# PREDICTION FUNCTION
# ----------------------------
def predict(file_path, model_path="voice_model.pth", classes_path="classes.npy"):
    # Load encoder classes
    classes = np.load(classes_path, allow_pickle=True)

    # Load model
    num_classes = len(classes)
    model = VoiceClassifier(num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Load audio
    signal, sr = librosa.load(file_path, sr=sample_rate)

    # Extract MFCCs
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
    mfcc = np.mean(mfcc.T, axis=0)

    # Convert to tensor
    features = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)  # add batch dimension

    # Predict
    with torch.no_grad():
        outputs = model(features)
        predicted_idx = torch.argmax(outputs, dim=1).item()

    predicted_speaker = classes[predicted_idx]
    print(f"\n🎤 Predicted Speaker: {predicted_speaker}")
    return predicted_speaker

# ----------------------------
# MAIN EXECUTION
# ----------------------------
if __name__ == "__main__":
    # Step 1: Train the model
    train_model()

    # Step 2: Ask user for voice input file
    test_file = input("\n👉 Enter path to a .wav file to identify the speaker: ").strip()

    if os.path.exists(test_file):
        predict(test_file)
    else:
        print("❌ File not found. Please give a valid .wav file path.")
