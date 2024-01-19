import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

class ChurnPredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Customer Churn Prediction App")
        self.df = None
        self.scaler = StandardScaler()
        self.model = None
        self.X_train = None
        self.y_train = None  # Initialize the 'y_train' attribute

        self.load_data_button = tk.Button(root, text="Load Dataset", command=self.load_data)
        self.load_data_button.pack(pady=10)

        self.train_model_button = tk.Button(root, text="Train Model", command=self.train_model)
        self.train_model_button.pack(pady=10)

        self.visualize_button = tk.Button(root, text="Visualize Results", command=self.visualize_results)
        self.visualize_button.pack(pady=10)

        self.result_text = tk.Text(root, height=10, width=50)
        self.result_text.pack(pady=10)

    def load_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        self.df = pd.read_csv(file_path)
        print("Columns in DataFrame:", self.df.columns)
        messagebox.showinfo("Success", "Dataset loaded successfully!")

    def train_model(self):
        if self.df is None:
            messagebox.showerror("Error", "Please load a dataset first.")
            return

        try:
            if 'Churn' not in self.df.columns:
                messagebox.showerror("Error", "'Churn' column not found in the dataset.")
                return

            X = self.df.drop('Churn', axis=1)

            # Handle categorical variables (assuming 'Churn' is the target column)
            categorical_cols = X.select_dtypes(include=['object']).columns
            X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
            X.fillna(0, inplace=True)

        except KeyError as e:
            messagebox.showerror("Error", f"Error in train_model: {e}")
            return

        y = self.df['Churn']
        self.X_train, _, self.y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

        X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.model = LogisticRegression(random_state=42)
        self.model.fit(X_train_scaled, self.y_train)
        messagebox.showinfo("Success", "Model trained successfully!")

    def visualize_results(self):
        if self.df is None:
            messagebox.showerror("Error", "Please load a dataset first.")
            return

        if self.model is None:
            messagebox.showerror("Error", "Please train the model first.")
            return

        # Use the trained model and scaler for visualization
        X_test = self.scaler.transform(self.X_train)
        predictions = self.model.predict(X_test)
        accuracy = accuracy_score(self.y_train, predictions)
        conf_matrix = confusion_matrix(self.y_train, predictions)
        classification_rep = classification_report(self.y_train, predictions)

        result_text = f'Accuracy: {accuracy}\n\nConfusion Matrix:\n{conf_matrix}\n\nClassification Report:\n{classification_rep}'

        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, result_text)

        # Visualize the confusion matrix
        plt.figure()
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Not Churn', 'Churn'],
                    yticklabels=['Not Churn', 'Churn'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    app = ChurnPredictionApp(root)
    root.mainloop()
