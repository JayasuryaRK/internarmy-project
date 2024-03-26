import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load your CSV file
file_path = 'dataset_small.csv'  # Replace with the path to your CSV file
data = pd.read_csv(file_path)

# Assuming your CSV file has a 'label' column and other feature columns
X = data.drop('phishing', axis=1)
y = data['phishing']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Support Vector Machine classifier
svm_classifier = SVC()

# Train the classifier
svm_classifier.fit(X_train, y_train)

# Make predictions on the test set
predictions = svm_classifier.predict(X_test)

# Evaluate the accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")

# Now, you can use the trained SVM classifier for making predictions on new data.
