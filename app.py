from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

app = Flask(__name__)

class ChurnPredictionApp:
    def __init__(self):
        self.df = None
        self.scaler = StandardScaler()
        self.model = None
        self.X_train = None
        self.y_train = None

    def load_data(self, file_path):
        self.df = pd.read_csv(file_path)

    def train_model(self):
        if self.df is None:
            return "Please load a dataset first."

        try:
            if 'Churn' not in self.df.columns:
                return "'Churn' column not found in the dataset."

            X = self.df.drop('Churn', axis=1)
            categorical_cols = X.select_dtypes(include=['object']).columns
            X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
            X.fillna(0, inplace=True)

        except KeyError as e:
            return f"Error in train_model: {e}"

        y = self.df['Churn']
        self.X_train, _, self.y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

        X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.model = LogisticRegression(random_state=42)
        self.model.fit(X_train_scaled, self.y_train)
        return "Model trained successfully!"

    def visualize_results(self):
        if self.df is None or self.model is None:
            return "Please load a dataset and train the model first."

        X_test = self.scaler.transform(self.X_train)
        predictions = self.model.predict(X_test)
        accuracy = accuracy_score(self.y_train, predictions)
        conf_matrix = confusion_matrix(self.y_train, predictions)
        classification_rep = classification_report(self.y_train, predictions)

        result_text = f'Accuracy: {accuracy}\n\nConfusion Matrix:\n{conf_matrix}\n\nClassification Report:\n{classification_rep}'

        # Visualize the confusion matrix
        plt.figure()
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Not Churn', 'Churn'],
                    yticklabels=['Not Churn', 'Churn'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')

        # Save the plot to a BytesIO object
        img_buf = BytesIO()
        plt.savefig(img_buf, format='png')
        img_buf.seek(0)
        img_data = base64.b64encode(img_buf.read()).decode('utf-8')
        plt.close()

        return result_text, img_data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/load_data', methods=['POST'])
def load_data():
    file_path = request.form['file_path']
    app_instance.load_data(file_path)
    return "Dataset loaded successfully!"

@app.route('/train_model')
def train_model():
    message = app_instance.train_model()
    return message

@app.route('/visualize_results')
def visualize_results():
    result_text, img_data = app_instance.visualize_results()
    return render_template('results.html', result_text=result_text, img_data=img_data)

if __name__ == '__main__':
    app_instance = ChurnPredictionApp()
    app.run(debug=False)

