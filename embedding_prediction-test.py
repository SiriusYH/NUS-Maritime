import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from tqdm import tqdm

# Define the same model architecture as in training
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

def load_components():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load the trained model
    model = SeverityClassifier(50)  # Using 50 as that's the PCA components used in training
    model.load_state_dict(torch.load('severity_classifier.pth'))
    model = model.to(device)
    model.eval()
    
    # Load PCA
    pca = pd.read_pickle('pca_model.pkl')
    
    # Load label encoder
    le = pd.read_pickle('label_encoder.pkl')
    
    return model, pca, le, device

def process_test_data(test_df, device):
    print("Loading BERT model...")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    bert_model = AutoModel.from_pretrained('bert-base-uncased', device_map="auto")
    
    def get_bert_embedding(text):
        inputs = tokenizer(str(text), 
                         return_tensors='pt', 
                         max_length=512, 
                         truncation=True, 
                         padding=True)
        
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = bert_model(**inputs)
        
        embeddings = outputs.last_hidden_state[:, 0, :]
        return embeddings.cpu().numpy()[0]
    
    # Generate embeddings for test data
    print("Generating BERT embeddings for test data...")
    embeddings = []
    for text in tqdm(test_df['def_text']):
        emb = get_bert_embedding(text)
        embeddings.append(emb)
    
    # Replicate the embedding to match the training data structure
    # (since training data had 5 text columns, we'll replicate the embedding 5 times)
    combined_embeddings = np.hstack([embeddings] * 5)
    
    return combined_embeddings

def predict_severity(test_df):
    # Load components
    model, pca, le, device = load_components()
    
    # Process test data
    print("Processing test data...")
    combined_embeddings = process_test_data(test_df, device)
    
    # Apply PCA transformation
    print("Applying PCA transformation...")
    reduced_embeddings = pca.transform(combined_embeddings)
    
    # Convert to tensor
    X_test_tensor = torch.FloatTensor(reduced_embeddings).to(device)
    
    # Make predictions
    print("Making predictions...")
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor)
        y_pred = torch.argmax(y_pred, dim=1).cpu().numpy()
    
    # Convert predictions to original labels
    predictions = le.inverse_transform(y_pred)
    
    # Add predictions to dataframe
    test_df['predicted_severity'] = predictions
    
    return test_df

if __name__ == "__main__":
    # Load test data
    print("Loading test data...")
    test_df = pd.read_csv('psc_severity_test.csv')
    test_df = test_df.dropna(subset=['def_text'])  # Only drop rows with missing text
    
    # Make predictions
    test_df = predict_severity(test_df)
    
    # Save results
    test_df.to_csv('test_predictions.csv', index=False)
    print("Predictions saved to 'test_predictions.csv'")