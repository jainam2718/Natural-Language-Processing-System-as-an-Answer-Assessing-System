import torch
import torch.nn as nn
import sent2vec
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



# Define neural network model
class SentenceSimilarityModel(nn.Module):
    def __init__(self, input_size):
        super(SentenceSimilarityModel, self).__init__()
        self.fc = nn.Linear(input_size, 1)

    def forward(self, x):
        output = torch.sigmoid(self.fc(x))  # Sigmoid for binary classification
        return output

# Custom dataset for pre-embedded sentences
class EmbeddingDataset(Dataset):
    def __init__(self, sentences1, sentences2, labels):
        self.sentences1 = sentences1
        self.sentences2 = sentences2
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sentences1[idx], self.sentences2[idx], self.labels[idx]

# Load the dataset from CSV
df = pd.read_csv("./data/biosses_back_translation_augmented.csv")

# Extract sentences and labels from the dataset
sentences1 = df["sentence1"]
sentences2 = df["sentence2"]
labels = df["score"] / 5  # Normalize the scores to be between 0 and 1

# Load the pre-trained Sent2Vec model
model_path = "./model/BioSentVec_PubMed_MIMICIII-bigram_d700.bin"
sent_model = sent2vec.Sent2vecModel()
try:
    sent_model.load_model(model_path)
except Exception as e:
    print(e)
print('Model successfully loaded')

# Pre-embed sentences using Sent2Vec
embeddings1 = [sent_model.embed_sentence(str(sentence)) for sentence in sentences1]
embeddings2 = [sent_model.embed_sentence(str(sentence)) for sentence in sentences2]

# Flatten embeddings and convert to tensors
flatten = lambda t: t.reshape(t.shape[0], -1)  # Define flatten function
embeddings1_tensor = flatten(torch.tensor(embeddings1, dtype=torch.float32))
embeddings2_tensor = flatten(torch.tensor(embeddings2, dtype=torch.float32))
labels_tensor = torch.tensor(labels.values, dtype=torch.float32)

# Create the dataset and dataloader
dataset = EmbeddingDataset(embeddings1_tensor, embeddings2_tensor, labels_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


# Initialize the model
input_size = embeddings1_tensor.size(1) + embeddings2_tensor.size(1)  # Calculate input size dynamically
model = SentenceSimilarityModel(input_size=input_size)
# Define optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for binary classification

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        embeddings1_batch, embeddings2_batch, labels_batch = batch
        concatenated_embeddings = torch.cat((embeddings1_batch, embeddings2_batch), dim=1)
        outputs = model(concatenated_embeddings)
        loss = criterion(outputs, labels_batch.unsqueeze(1))  # Ensure labels shape matches outputs
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')

# Load the dataset from CSV
df = pd.read_csv("./data/biosses_dataset.csv")

# Extract sentences and labels from the dataset
sentences1 = df["sentence1"]
sentences2 = df["sentence2"]
labels = df["score"] / 5  # Normalize the scores to be between 0 and 1

# Pre-embed sentences using Sent2Vec
embeddings1 = [sent_model.embed_sentence(str(sentence)) for sentence in sentences1]
embeddings2 = [sent_model.embed_sentence(str(sentence)) for sentence in sentences2]

# Flatten embeddings and convert to tensors
flatten = lambda t: t.reshape(t.shape[0], -1)  # Define flatten function
embeddings1_tensor = flatten(torch.tensor(embeddings1, dtype=torch.float32))
embeddings2_tensor = flatten(torch.tensor(embeddings2, dtype=torch.float32))
labels_tensor = torch.tensor(labels.values, dtype=torch.float32)

# Create the dataset and dataloader
dataset = EmbeddingDataset(embeddings1_tensor, embeddings2_tensor, labels_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Evaluation
model.eval()
predictions = []
true_labels = []
with torch.no_grad():
    for embeddings1_batch, embeddings2_batch, labels_batch in dataloader:
        concatenated_embeddings = torch.cat((embeddings1_batch, embeddings2_batch), dim=1)
        outputs = model(concatenated_embeddings)
        predictions.extend(outputs.squeeze().tolist())
        true_labels.extend(labels_batch.tolist())



true_labels = [1 if pred >= 0.5 else 0 for pred in true_labels]
for t in range(0, 10):
    threshold = t/10
    # Convert predictions to binary (0 or 1) based on threshold
    binary_predictions = [1 if pred >= threshold else 0 for pred in predictions]

    # Calculate evaluation metrics
    accuracy = accuracy_score(true_labels, binary_predictions)
    precision = precision_score(true_labels, binary_predictions)
    recall = recall_score(true_labels, binary_predictions)
    f1 = f1_score(true_labels, binary_predictions)
    print(threshold)
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')

