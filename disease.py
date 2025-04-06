import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pickle
from flask import Flask, request, jsonify

try:
    train_df = pd.read_csv("datasets/Training.csv")
    test_df = pd.read_csv("datasets/Testing.csv")
except FileNotFoundError:
    print("Error: Training.csv or Testing.csv not found. Please check the file paths.")

train_df.fillna(0, inplace=True)
test_df.fillna(0, inplace=True)

X_train = train_df.drop(columns=['prognosis'])
y_train = train_df['prognosis']

X_test = test_df.drop(columns=['prognosis'])
y_test = test_df['prognosis']

le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

symptom_columns = X_train.columns.tolist()

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train_encoded)

accuracy = accuracy_score(y_test_encoded, model.predict(X_test))
print(f"Symptoms Model Accuracy: {accuracy * 100:.2f}%")

from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X_train, y_train_encoded, cv=5)
print(f"Cross-validation scores: {scores.mean() * 100:.2f}% (+/- {scores.std() * 2 * 100:.2f}%)")

try:
    with open("disease_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
except Exception as e:
    print(f"Error saving models: {e}")

try:
    descriptions_df = pd.read_csv("datasets/symptom_Description.csv")
    precautions_df = pd.read_csv("datasets/symptom_precaution.csv")
except FileNotFoundError:
    print("Error: descriptions.csv or precautions.csv not found. Please check the file paths.")

app = Flask(__name__)

@app.route('/predict_symptoms', methods=['POST'])
def predict_symptoms():
    try:
        data = request.json
        if not data or 'symptoms' not in data:
            return jsonify({"error": "No symptoms provided in the request"}), 400

        user_symptoms = data['symptoms']

        symptom_vector = [1 if sym in user_symptoms else 0 for sym in symptom_columns]
        symptoms = np.array(symptom_vector).reshape(1, -1)

        prediction = model.predict(symptoms)[0]
        disease = le.inverse_transform([prediction])[0]

        description = descriptions_df[descriptions_df['Disease'] == disease]['Description'].values[0]
        precautions = precautions_df[precautions_df['Disease'] == disease][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']].values.tolist()[0]

        return jsonify({
            "predicted_disease": disease,
            "description": description,
            "precautions": precautions
        })
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    try:
        app.run(debug=True, host='0.0.0.0', port=5001)
    except KeyboardInterrupt:
        print("Server stopped by user")
    except Exception as e:
        print(f"Error running Flask server: {e}")
