import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from data.dataset import AcousticDataset
from models.cnn import AcousticCNN

def evaluate_model(data_dir, model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load entire dataset (typically you'd have a separate test set folder)
    test_dataset = AcousticDataset(data_dir=data_dir)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    if len(test_dataset) == 0:
        print("Dataset is empty.")
        return
        
    model = AcousticCNN(num_classes=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Calculate performance metrics
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    
    print("\n--- Evaluation Metrics ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Normal', 'Anomaly']))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data', help='Directory containing test normal & anomaly subfolders')
    parser.add_argument('--model_path', type=str, default='checkpoints/acoustic_cnn_latest.pth', help='Path to weights')
    args = parser.parse_args()
    evaluate_model(data_dir=args.data_dir, model_path=args.model_path)
