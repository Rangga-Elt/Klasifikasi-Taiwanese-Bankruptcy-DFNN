import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt

from google.colab import files
from imblearn.combine import SMOTEENN
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

# Upload Dataset dari Device
uploaded = files.upload()
file_name = list(uploaded.keys())[0]

# Load Dataset
def load_data(file_name):
    if file_name.endswith('.csv'):
        return pd.read_csv(file_name)
    elif file_name.endswith('.xlsx'):
        return pd.read_excel(file_name)
    else:
        raise ValueError("Unsupported file format. Please use CSV or XLSX.")

data = load_data(file_name)

# Preprocessing
# Hapus fitur yang nilainya 0 semua, 1 semua, atau 2 semua
data = data.loc[:, (data != data.iloc[0]).any()]

# Pisahkan fitur dan label target
X = data.drop(columns=['Bankrupt?'])  
y = data['Bankrupt?']  

# Buat fitur baru jika ada fitur yang nilainya 0
for col in X.columns: 
    if (X[col] == 0).any():
        X[f'{col}_zero_flag'] = (X[col] == 0).astype(int)

# Stratified K-Fold Cross Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Placeholder untuk menyimpan hasil
all_y_true = []  
all_y_pred = []  
all_histories = [] 

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"Fold {fold + 1}")

    # Split data asli menjadi training dan validation
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Oversampling menggunakan SMOTE hanya pada data training
    smote_enn = SMOTEENN(random_state=42)
    X_train_resampled, y_train_resampled = smote_enn.fit_resample(X_train, y_train)

    # Tambahkan fitur flag untuk menandai data SMOTE
    X_train_resampled['smote_flag'] = 1
    X_train_with_flag = pd.concat([X_train.assign(smote_flag=0), X_train_resampled], axis=0)
    y_train_with_flag = pd.concat([y_train, y_train_resampled], axis=0)

    # Tambahkan kolom 'smote_flag' 
    X_val['smote_flag'] =

    # StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_with_flag)
    X_val_scaled = scaler.transform(X_val)

    # Model Deep Learning
    def create_model(input_dim):
        model = Sequential([
            Dense(16, activation='relu', input_dim=input_dim),
            BatchNormalization(),
            Dropout(0.2),
            Dense(16, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])
        return model

    # Latih Model
    model = create_model(input_dim=X_train_scaled.shape[1])
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train_with_flag), y=y_train_with_flag)
    class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}  

    # Fungsi Weighted Binary Cross-Entropy Loss
    def weighted_binary_crossentropy(y_true, y_pred):
      weights = tf.gather(class_weights, tf.cast(y_true, tf.int32))
      loss = tf.keras.backend.binary_crossentropy(y_true, y_pred)
      weighted_loss = tf.cast(weights, tf.float32) * loss

      return tf.keras.backend.mean(weighted_loss)

    model.compile(optimizer=Adam(learning_rate=0.0005),
              loss=weighted_binary_crossentropy, 
              metrics=['accuracy'])

    history = model.fit(
      X_train_scaled, y_train_with_flag,
      validation_data=(X_val_scaled, y_val),
      epochs=100,
      batch_size=32,
      callbacks=[early_stopping],
      verbose=1
    )
    all_histories.append(history)

    # Evaluasi model menggunakan validation yang belum di-SMOTE
    y_pred = (model.predict(X_val_scaled) > 0.2).astype(int)
    all_y_true.extend(y_val)
    all_y_pred.extend(y_pred.flatten())

    # Classification Report
    report = classification_report(y_val, y_pred, output_dict=True)
    print(f"Classification Report Fold {fold + 1}:\n", classification_report(y_val, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_val, y_pred)
    print(f"Confusion Matrix Fold {fold + 1}:\n", cm)

    # Plot Loss Curve
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Loss Curve - Fold {fold + 1}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Plot Learning Curve
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'Learning Curve - Fold {fold + 1}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

# Gabungkan semua prediksi untuk membuat classification report dan confusion matrix global
global_classification_report = classification_report(all_y_true, all_y_pred, output_dict=True)
print("Global Classification Report:\n", classification_report(all_y_true, all_y_pred))

global_confusion_matrix = confusion_matrix(all_y_true, all_y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(global_confusion_matrix, fmt='.0f', annot=True, cmap='Blues',
            xticklabels=['Not Bankrupt', 'Bankrupt'],
            yticklabels=['Not Bankrupt', 'Bankrupt'])
plt.title('Global Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Validation Curve
plt.figure(figsize=(10, 6))
min_epochs = min(len(history.history['accuracy']) for history in all_histories)

train_accuracies = np.mean([[history.history['accuracy'][i] for history in all_histories if i < len(history.history['accuracy'])] for i in range(min_epochs)], axis=1)
val_accuracies = np.mean([[history.history['val_accuracy'][i] for history in all_histories if i < len(history.history['val_accuracy'])] for i in range(min_epochs)], axis=1)

plt.plot(train_accuracies, label='Average Train Accuracy')
plt.plot(val_accuracies, label='Average Validation Accuracy')
plt.title('Validation Curve (Average Across Folds)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Learning Curve
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title(f'Learning Curve - Fold {fold + 1}')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
