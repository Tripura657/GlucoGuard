import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pickle

# Load dataset
data = pd.read_csv('diabetes_risk_prediction_dataset.csv')

# Define features and label
features = ['Polyuria', 'Gender', 'Polydipsia', 'sudden weight loss',
            'partial paresis', 'visual blurring', 'Alopecia', 'Irritability']
X = data[features]
y = data['class']

# Initialize encoder dictionary
encoder_dict = {}

# Encode each feature using LabelEncoder and store the encoders
for col in features:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    encoder_dict[col] = le  # Save the encoder

# Print Gender encoding for verification
gender_encoder = encoder_dict['Gender']
print("✅ Gender classes:", list(gender_encoder.classes_))
print("   'Female' encoded as:", gender_encoder.transform(['Female'])[0])
print("   'Male' encoded as:", gender_encoder.transform(['Male'])[0])

# Encode target
label_encoder_y = LabelEncoder()
y = label_encoder_y.fit_transform(y)

# Save feature encoders
with open('encoder.pkl', 'wb') as f:
    pickle.dump(encoder_dict, f)

# Save target encoder
with open('target_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder_y, f)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Calculate class weights to handle imbalance
class_weights = compute_class_weight(class_weight='balanced',
                                     classes=np.unique(y_train),
                                     y=y_train)
class_weight_dict = dict(enumerate(class_weights))
print("📊 Computed class weights:", class_weight_dict)

# Build the model
model = Sequential()
model.add(Dense(16, input_dim=len(features), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model with class weights
model.fit(X_train, y_train, epochs=100, batch_size=10,
          validation_data=(X_test, y_test),
          class_weight=class_weight_dict)

# Save the trained model
model.save('diabetes_model.h5')

print("✅ Model trained and saved successfully with class balancing!")