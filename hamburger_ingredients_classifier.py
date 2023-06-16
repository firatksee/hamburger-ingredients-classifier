import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import SGD
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
import seaborn as sns
import random


class Model():
    def __init__(self):
        # Define the directories
        self.current_directory = os.getcwd()
        self.data_directory = os.path.join(self.current_directory, "data")
        self.model_directory = os.path.join(self.current_directory, "model")
        # Define the list of ingredients
        self.ingredients = os.listdir(self.data_directory)
        # Load the data
        self.data, self.labels = self.read_data(
            self.data_directory, self.ingredients)
        # Split train data and test data
        self.train_data, self.test_data, self.train_labels, self.test_labels = self.split_and_normalize_data(
            self.data, self.labels)
        # Train the model
        self.model, self.history = self.train_model(
            self.train_data, self.test_data, self.train_labels, self.test_labels)
        # Save the model for further use
        self.save_model(self.model, self.model_directory)
        # Evaluate and display the stats of the model
        self.evaluate_model(self.model, self.test_data, self.test_labels)
        self.display_stats()

    def read_data(self, data_dir, ingredients):
        # Collect each data and its corresponding label
        data = []
        labels = []
        for ingredient in ingredients:
            ingredient_dir = os.path.join(data_dir, ingredient)
            for filename in os.listdir(ingredient_dir):
                try:
                    img_path = os.path.join(ingredient_dir, filename)
                    image = cv2.imread(img_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = cv2.resize(image, (100, 100))
                    data.append(image)
                    labels.append(ingredient)
                except:
                    continue

        # Convert the data and labels to numpy arrays
        data = np.array(data)
        labels = np.array(labels)

        return data, labels

    def split_and_normalize_data(self, data, labels):
        # Perform train-test split
        train_data, test_data, train_labels, test_labels = train_test_split(
            data, labels, test_size=0.2, random_state=42)

        # Normalize the pixel values to the range [0, 1]
        train_data = train_data.astype('float32') / 255.0
        test_data = test_data.astype('float32') / 255.0

        # Perform one-hot encoding on the labels
        self.le = LabelEncoder()
        train_labels = self.le.fit_transform(train_labels)
        test_labels = self.le.transform(test_labels)
        self.num_classes = len(self.le.classes_)
        train_labels = to_categorical(train_labels, self.num_classes)
        test_labels = to_categorical(test_labels, self.num_classes)

        return train_data, test_data, train_labels, test_labels

    def train_model(self, train_data, test_data, train_labels, test_labels):
        # Define the LeNet-5 model architecture
        model = Sequential()
        model.add(Conv2D(6, (5, 5), activation='relu',
                  input_shape=(100, 100, 3)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(16, (5, 5), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(120, activation='relu'))
        model.add(Dense(84, activation='relu'))
        model.add(Dense(self.num_classes, activation='softmax'))

        # Compile the model
        optimizer = SGD(learning_rate=0.01)
        model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy', metrics=['accuracy'])

        # Train the model
        history = model.fit(train_data, train_labels, validation_data=(
            test_data, test_labels), epochs=20, batch_size=32)

        return model, history

    def save_model(self, model, path):
        # Save the model
        try:
            model.save(os.path.join(path, "model.h5"))
        except:
            print(
                "[WARNING] Model could not be saved.")

    def evaluate_model(self, model, test_data, test_labels):
        # Evaluate the model
        predictions = model.predict(test_data)
        print(classification_report(np.argmax(test_labels, axis=1),
                                    np.argmax(predictions, axis=1), target_names=self.le.classes_))

    def display_stats(self):
        # Define and prepare the window for stats and plots
        fig = plt.figure(figsize=(20, 8))
        grid_size = (2, 8)
        ax1 = plt.subplot2grid(grid_size, (0, 0), colspan=2)
        ax2 = plt.subplot2grid(grid_size, (0, 2), colspan=2)
        ax3 = plt.subplot2grid(grid_size, (0, 4), colspan=2)
        ax4 = plt.subplot2grid(grid_size, (0, 6), colspan=2)
        second_row = [plt.subplot2grid(grid_size, (1, i)) for i in range(8)]

        # Plot the training loss
        ax1.plot(self.history.history['loss'], label='Train Loss')
        ax1.plot(self.history.history['val_loss'], label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()

        # Plot the training accuracy
        ax2.plot(self.history.history['accuracy'], label='Train Accuracy')
        ax2.plot(self.history.history['val_accuracy'],
                 label='Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()

        # Confusion Matrix - Heatmap
        predictions = self.model.predict(self.test_data)
        predicted_labels = np.argmax(predictions, axis=1)
        cm = confusion_matrix(
            np.argmax(self.test_labels, axis=1), predicted_labels)

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.ingredients, yticklabels=self.ingredients, ax=ax3)
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, fontsize=8)
        ax3.set_yticklabels(ax3.get_yticklabels(), rotation=45, fontsize=8)
        ax3.set_xlabel('Predicted Labels')
        ax3.set_ylabel('True Labels')
        ax3.set_title('Confusion Matrix')

        # Class Distribution - Bar chart
        ingredient_counts = np.unique(self.labels, return_counts=True)
        ax4.bar(self.ingredients, ingredient_counts[1])
        ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, fontsize=8)
        ax4.set_xlabel('Ingredients')
        ax4.set_ylabel('Count')
        ax4.set_title('Class Distribution')

        # Sample Predictions
        num_samples = 8
        sample_indices = random.sample(range(len(self.test_data)), num_samples)
        sample_images = self.test_data[sample_indices]
        sample_labels = self.test_labels[sample_indices]

        sample_predictions = self.model.predict(sample_images)
        predicted_classes = np.argmax(sample_predictions, axis=1)

        for i in range(num_samples):
            ax = second_row[i]
            ax.imshow(sample_images[i])
            ax.axis('off')
            ax.set_title(
                f'Pred: {self.ingredients[predicted_classes[i]]}\nTrue: {self.ingredients[np.argmax(sample_labels[i])]}')

        # Display the stats window
        plt.subplots_adjust(wspace=1)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    Model()
