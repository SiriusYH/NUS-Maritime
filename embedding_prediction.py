import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.decomposition import PCA
from tqdm import tqdm
import os



# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Load data
print("Loading data...")
df = pd.read_csv('train.csv')
df = df.dropna()

# Process each text column separately
text_columns = ['finding', 'description', 'immediate_causes', 
                'root_cause', 'corrective_action']

if os.path.exists("combined_embeddings.npy"):
    print("Loading combined embeddings...")
    combined_embeddings = np.load('combined_embeddings.npy')
    embeddings_dict = {}
    for column in text_columns:
        embeddings_dict[column] = np.load(f'{column}_embeddings.npy')
else:
    # Initialize BERT tokenizer and model
    print("Loading BERT model...")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModel.from_pretrained('bert-base-uncased', device_map = "auto")

    # Function to get BERT embeddings
    def get_bert_embedding(text):
        # Tokenize and encode text
        inputs = tokenizer(str(text), 
                        return_tensors='pt', 
                        max_length=512, 
                        truncation=True, 
                        padding=True)
        
        # Move inputs to GPU
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get BERT output
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Use [CLS] token embedding (first token)
        embeddings = outputs.last_hidden_state[:, 0, :]
        return embeddings.cpu().numpy()[0]
    print("Generating BERT embeddings for each text column...")
    embeddings_dict = {}
    for column in tqdm(text_columns):
        embeddings = []
        for text in df[column]:
            emb = get_bert_embedding(text)
            embeddings.append(emb)
        embeddings_dict[column] = np.array(embeddings)

    # Combine embeddings for each text column
    print("Combining embeddings...")
    combined_embeddings = np.hstack([embeddings_dict[col] for col in text_columns])

    # Save embeddings for future use
    print("Saving embeddings...")
    np.save('combined_embeddings.npy', combined_embeddings)
    for column in text_columns:
        np.save(f'{column}_embeddings.npy', embeddings_dict[column])

# Dimensionality reduction for visualization and analysis
print("Performing dimensionality reduction...")
pca = PCA(n_components=50)
reduced_embeddings = pca.fit_transform(combined_embeddings)

# Prepare for modeling
print("Preparing for modeling...")
le = LabelEncoder()
y = le.fit_transform(df['annotation_severity'])
print(df['annotation_severity'].value_counts())

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    reduced_embeddings, y, test_size=0.2, random_state=42
)


# Train a simple neural network
print("Training neural network...")
class SeverityClassifier(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layer1 = torch.nn.Linear(input_dim, 512)
        self.bn1 = torch.nn.BatchNorm1d(512)
        self.dropout1 = torch.nn.Dropout(0.3)
        
        self.layer2 = torch.nn.Linear(512, 256)
        self.bn2 = torch.nn.BatchNorm1d(256)
        self.dropout2 = torch.nn.Dropout(0.2)
        
        self.layer3 = torch.nn.Linear(256, 64)
        self.bn3 = torch.nn.BatchNorm1d(64)
        self.dropout3 = torch.nn.Dropout(0.1)
        
        self.output = torch.nn.Linear(64, 4)
        
    def forward(self, x):
        x = self.bn1(F.gelu(self.layer1(x)))
        x = self.dropout1(x)
        x = self.bn2(F.gelu(self.layer2(x)))
        x = self.dropout2(x)
        x = self.bn3(F.gelu(self.layer3(x)))
        x = self.dropout3(x)
        x = self.output(x)
        return x

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train).to(device)
y_train_tensor = torch.LongTensor(y_train).to(device)
X_test_tensor = torch.FloatTensor(X_test).to(device)

print("Unique labels in y_train:", torch.unique(y_train_tensor))
print("Min label:", torch.min(y_train_tensor).item())
print("Max label:", torch.max(y_train_tensor).item())

# Initialize model and training parameters
model = SeverityClassifier(reduced_embeddings.shape[1]).to(device)
criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters())

# Training loop
print("Training model...")
n_epochs = 200
batch_size = 32
for epoch in range(n_epochs):
    model.train()
    total_loss = 0
    num_batches = 0
    
    for i in range(0, len(X_train), batch_size):
        batch_X = X_train_tensor[i:i+batch_size]
        batch_y = y_train_tensor[i:i+batch_size]
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    if (epoch + 1) % 2 == 0:
        print(f'Epoch [{epoch+1}/{n_epochs}], Average Loss: {avg_loss:.4f}')

# Make predictions
print("Making predictions...")
model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor)
    y_pred = torch.argmax(y_pred, dim=1).cpu().numpy()
    
def plot_confusion_matrix(y_true, y_pred, class_names):
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure and axes
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    
    # Add labels
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Show plot
    plt.show()

# Convert predictions back to original labels
y_test_original = le.inverse_transform(y_test)
y_pred_original = le.inverse_transform(y_pred)

# Get unique class names in order
class_names = le.classes_

# Plot confusion matrix
plot_confusion_matrix(y_test_original, y_pred_original, class_names)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, 
                          target_names=le.classes_))

# Analyze embeddings
print("\nAnalyzing embeddings...")

# 1. Visualize clustering of severity levels
plt.figure(figsize=(10, 6))
pca_viz = PCA(n_components=2)
viz_embeddings = pca_viz.fit_transform(combined_embeddings)

plt.scatter(viz_embeddings[:, 0], viz_embeddings[:, 1], 
           c=y, cmap='viridis', alpha=0.5)
plt.title('BERT Embeddings Visualization by Severity')
plt.colorbar(label='Severity Level')
plt.show()

# 2. Analyze semantic similarity between different text columns
print("\nSemantic Similarity Analysis:")
for col1 in text_columns:
    for col2 in text_columns:
        if col1 < col2:
            similarity = np.mean([
                F.cosine_similarity(
                    torch.FloatTensor(emb1.reshape(1, -1)).to(device),
                    torch.FloatTensor(emb2.reshape(1, -1)).to(device)
                ).cpu().item()
                for emb1, emb2 in zip(embeddings_dict[col1], 
                                    embeddings_dict[col2])
            ])
            print(f"{col1} - {col2}: {similarity:.3f}")

# 3. Analyze contribution of each text column
print("\nColumn Importance Analysis:")
for column in text_columns:
    # Train a simple model using only this column
    column_embeddings = embeddings_dict[column]
    pca_col = PCA(n_components=10)
    reduced_col = pca_col.fit_transform(column_embeddings)
    
    X_train_col, X_test_col, y_train_col, y_test_col = train_test_split(
        reduced_col, y, test_size=0.2, random_state=42
    )
    
    # Simple linear model for each column
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_col, y_train_col)
    score = clf.score(X_test_col, y_test_col)
    print(f"{column}: Accuracy = {score:.3f}")

# 4. Find similar deficiencies
print("\nSimilar Deficiency Analysis:")
from sklearn.metrics.pairwise import cosine_similarity

def find_similar_deficiencies(idx, top_k=5):
    query_embedding = combined_embeddings[idx]
    similarities = cosine_similarity(
        query_embedding.reshape(1, -1), 
        combined_embeddings
    )[0]
    
    # Get top k similar indices (excluding self)
    similar_indices = np.argsort(similarities)[::-1][1:top_k+1]
    return similar_indices, similarities[similar_indices]

# Example: Find similar deficiencies for a high-severity case
high_severity_idx = np.where(y == 2)[0][0]  # First high-severity case
similar_idx, similarities = find_similar_deficiencies(high_severity_idx)

print("\nExample Similar Deficiencies:")
print("Original Deficiency:")
print(df.iloc[high_severity_idx]['finding'])
print("\nSimilar Deficiencies:")
for idx, sim in zip(similar_idx, similarities):
    print(f"\nSimilarity: {sim:.3f}")
    print(f"Finding: {df.iloc[idx]['finding']}")
    print(f"Severity: {df.iloc[idx]['annotation_severity']}")
    
# Save the trained model
torch.save(model.state_dict(), 'severity_classifier.pth')

# Save the PCA model
import pickle
with open('pca_model.pkl', 'wb') as f:
    pickle.dump(pca, f)

# Save the label encoder
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)