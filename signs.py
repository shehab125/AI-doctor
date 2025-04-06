# استيراد المكتبات اللازمة
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import random
import joblib

# ضبط البذور العشوائية لضمان التكرارية
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# تحميل واكتشاف البيانات
df = pd.read_csv("datasets/human_vital_signs_dataset_2024.csv")

# معالجة البيانات مسبقًا
# ترميز عمود الجنس
label_encoder_gender = LabelEncoder()
df['Gender'] = label_encoder_gender.fit_transform(df['Gender'])  # 0 للذكور، 1 للإناث

# تحديد الميزات والهدف
features = ['Heart Rate', 'Respiratory Rate', 'Body Temperature', 'Oxygen Saturation', 
            'Systolic Blood Pressure', 'Diastolic Blood Pressure', 'Age', 'Gender', 'Weight (kg)', 'Height (m)']
X = df[features]
y = df['Risk Category']

# ترميز الفئات المستهدفة (High Risk -> 1, Low Risk -> 0)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# تقسيم البيانات إلى تدريب، تحقق، واختبار
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=seed)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=seed)

# توحيد المقاييس للميزات
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# تحويل البيانات إلى تينسور
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# إنشاء DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# تعريف نموذج الشبكة العصبية بأحجام طبقات معدلة
class VitalSignsModel(nn.Module):
    def __init__(self, input_size):
        super(VitalSignsModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# تهيئة النموذج
input_size = X_train.shape[1]
model = VitalSignsModel(input_size)

# تعريف الخسارة والمحسن
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# حلقة التدريب مع التحقق
num_epochs = 50
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        predicted = (torch.sigmoid(outputs) >= 0.5).float()
        correct_train += (predicted == labels).sum().item()
        total_train += labels.size(0)
    
    train_loss = running_loss / len(train_loader)
    train_accuracy = correct_train / total_train
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    
    # التحقق
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            predicted = (torch.sigmoid(outputs) >= 0.5).float()
            correct_val += (predicted == labels).sum().item()
            total_val += labels.size(0)
    
    val_loss /= len(val_loader)
    val_accuracy = correct_val / total_val
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)
    
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

# رسم منحنيات الخسارة والدقة
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss over Epochs')

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy over Epochs')
plt.show()

# تقييم النموذج على مجموعة الاختبار
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    test_probs = torch.sigmoid(test_outputs).numpy().flatten()
    test_predicted_classes = (test_probs >= 0.5).astype(int)
    test_true_classes = y_test

# حساب المقاييس
accuracy = accuracy_score(test_true_classes, test_predicted_classes)
precision = precision_score(test_true_classes, test_predicted_classes)
recall = recall_score(test_true_classes, test_predicted_classes)
f1 = f1_score(test_true_classes, test_predicted_classes)
roc_auc = roc_auc_score(test_true_classes, test_probs)

print(f"Test Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

# رسم منحنى ROC
fpr, tpr, _ = roc_curve(test_true_classes, test_probs)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

# اختبار النموذج مع بيانات يدوية
manual_data = [
    [99, 16, 36.654748, 95.011801, 118, 72, 41, 0, 96.006188, 1.833629],
    [83, 12, 36.044191, 98.584497, 111, 84, 50, 0, 79.295332, 1.672735],
    [79, 12, 36.884979, 95.987129, 130, 70, 22, 1, 79.869933, 1.922334],
    [66, 15, 36.957178, 97.916267, 131, 77, 61, 1, 53.923400, 1.896381],
    [72, 16, 36.8, 98, 120, 80, 20, 1, 78, 1.78]
]

manual_data = np.array(manual_data)
manual_data_scaled = scaler.transform(manual_data)
manual_data_tensor = torch.tensor(manual_data_scaled, dtype=torch.float32)

model.eval()
with torch.no_grad():
    outputs = model(manual_data_tensor)
    predicted_probs = torch.sigmoid(outputs).numpy().flatten()
    predicted_classes = (predicted_probs >= 0.5).astype(int)

predicted_risk_categories = label_encoder.inverse_transform(predicted_classes)
results_df = pd.DataFrame({
    "Data Point": range(1, len(manual_data) + 1),
    "Predicted Risk Category": predicted_risk_categories
})
print(results_df)

# حفظ النموذج، المقياس، والمشفر
model_save_path = "human_vital_sign_model.pth"
torch.save(model.state_dict(), model_save_path)

scaler_save_path = "scaler.pkl"
joblib.dump(scaler, scaler_save_path)

label_encoder_save_path = "label_encoder.pkl"
joblib.dump(label_encoder, label_encoder_save_path)