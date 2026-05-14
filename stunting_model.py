import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 1. Load the data
df = pd.read_csv('stunting_wasting_dataset.csv')

# 2. Preprocess the data
# Convert Gender (Jenis Kelamin) to numbers: Laki-laki -> 0, Perempuan -> 1
le_gender = LabelEncoder()
df['Jenis Kelamin'] = le_gender.fit_transform(df['Jenis Kelamin'])

# Features (X) are the inputs; Target (y) is what we want to predict
X = df[['Jenis Kelamin', 'Umur (bulan)', 'Tinggi Badan (cm)', 'Berat Badan (kg)']]
y = df['Wasting']

# 3. Split the data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Initialize and Train the AI Model (Random Forest)
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# 5. Test the AI
predictions = model.predict(X_test)

# 6. See how well it did
print(f"Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")
print("\nDetailed Report:")
print(classification_report(y_test, predictions))

# 7. Make a manual prediction
# Example: Laki-laki (0), 15 months old, 80cm tall, 10kg weight
sample_child = [[0, 15, 80.0, 10.0]]
result = model.predict(sample_child)
print(f"\nPrediction for sample child: {result[0]}")

# Generate the confusion matrix
cm = confusion_matrix(y_test, predictions, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)

# Plot it
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix: Stunting Prediction")
plt.savefig("confusion_matrix.png")
print("Confusion Matrix saved as confusion_matrix.png")