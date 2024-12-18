import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Paths to your icon images (10 images per class)
# Organize them into separate folders for each icon class
icon_classes = ["bugu", "naiman", "sayak"]
data_path = "dataset/"  # e.g., "icons/"

# Prepare data and labels
X = []  # Feature list
y = []  # Labels

for label, icon_class in enumerate(icon_classes):
    for i in range(1, 6):  # Assuming 10 images per class
        image_path = f"{data_path}/{icon_class}/image{i}.jpg"
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image_resized = cv2.resize(image, (50, 50))  # Resize to 50x50
        X.append(image_resized.flatten())  # Flatten the image into a 1D vector
        y.append(label)  # Label the image (0 for icon1, 1 for icon2, etc.)

# Convert to NumPy arrays
X = np.array(X)
y = np.array(y)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train an SVM classifier
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# Test the classifier
y_pred = clf.predict(X_test)
print("Accuracy of testing:", accuracy_score(y_test, y_pred))

# Recognize a new image
def predict_icon(image_path, model):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_resized = cv2.resize(image, (50, 50))
    features = image_resized.flatten().reshape(1, -1)
    prediction = model.predict(features)
    return icon_classes[prediction[0]], image

# Function to display the image with prediction
def display_prediction(image_path, model):
    prediction, _ = predict_icon(image_path, model)
    image_colored = cv2.imread(image_path)  # Load colored image for display

    if image_colored is None:
        print(f"Unable to load image: {image_path}")
        return

    # Resize image for smaller display
    scale_percent = 50  # Percent of original size
    width = int(image_colored.shape[1] * scale_percent / 100)
    height = int(image_colored.shape[0] * scale_percent / 100)
    image_colored = cv2.resize(image_colored, (width, height))

    # Add a rectangle and label
    x, y, w, h = 10, 10, width - 20, height - 20  # Bounding box
    cv2.rectangle(image_colored, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(image_colored, prediction, (x + 10, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the image
    cv2.imshow("Icon Recognition", image_colored)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example: Predict and display a new image
new_image_path = "test_image2.jpg"
display_prediction(new_image_path, clf)