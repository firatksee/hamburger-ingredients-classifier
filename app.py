import sys
import os
import cv2
import io
import datetime
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QSpinBox, QRadioButton
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from keras.models import load_model
from hamburger_ingredients_classifier import Model


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # Define the model for further assignment
        self.model = None
        # Define the directories
        self.current_directory = os.getcwd()
        self.data_directory = os.path.join(self.current_directory, "data")
        # Define the list of ingredients
        self.ingredients = os.listdir(self.data_directory)
        # Initialize variables
        self.image_path = None
        self.grid_images = []
        self.predictions = []
        # Create the widgets
        self.canvas_width = 800
        self.canvas_height = self.canvas_width
        self.create_widgets()

    def create_widgets(self):
        self.setWindowTitle("Hamburger Ingredient Classifier")

        main_widget = QWidget(self)
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        canvas_layout = QVBoxLayout(self)

        # Create input canvas
        self.input_label = QLabel(self)
        self.input_label.setFixedSize(
            self.canvas_width, self.canvas_height // 2)
        self.input_label.setAlignment(Qt.AlignCenter)
        canvas_layout.addWidget(self.input_label)

        # Create output canvas
        self.output_label = QLabel(self)
        self.output_label.setFixedSize(
            self.canvas_width, self.canvas_height // 2)
        self.output_label.setAlignment(Qt.AlignCenter)
        canvas_layout.addWidget(self.output_label)

        input_layout = QVBoxLayout(self)
        input_layout.setAlignment(Qt.AlignTop)
        input_layout.addSpacing(32)

        # Create radio button for model training option
        self.radio_button1 = QRadioButton("Train Model", self)
        self.radio_button1.clicked.connect(self.radio_button_clicked)
        input_layout.addWidget(self.radio_button1)

        # Create radio button for model reading option
        self.radio_button2 = QRadioButton("Read Model", self)
        self.radio_button2.clicked.connect(self.radio_button_clicked)
        input_layout.addWidget(self.radio_button2)

        input_layout.addSpacing(8)

        # Create button for model loading
        self.load_model_button = QPushButton("Load Model", self)
        self.load_model_button.clicked.connect(self.load_model_btn)
        self.load_model_button.setEnabled(False)
        input_layout.addWidget(self.load_model_button)

        input_layout.addSpacing(32)

        # Create number input are for defining number of rows
        self.rows_label = QLabel("Number of rows:", self)
        input_layout.addWidget(self.rows_label)

        self.rows_input = QSpinBox(self)
        self.rows_input.setValue(2)
        self.rows_input.setRange(1, 5)
        input_layout.addWidget(self.rows_input)

        input_layout.addSpacing(16)

        # Create number input are for defining number of columns
        self.columns_label = QLabel("Number of Columns:", self)
        input_layout.addWidget(self.columns_label)

        self.columns_input = QSpinBox(self)
        self.columns_input.setValue(5)
        self.columns_input.setRange(1, 5)
        input_layout.addWidget(self.columns_input)

        input_layout.addSpacing(32)

        # Create button for browsing input image
        self.browse_button = QPushButton("Browse Image", self)
        self.browse_button.clicked.connect(self.browse_image)
        self.browse_button.setEnabled(False)
        input_layout.addWidget(self.browse_button)

        # Create button for predicting the input
        self.predict_button = QPushButton("Predict", self)
        self.predict_button.clicked.connect(self.predict_image)
        self.predict_button.setEnabled(False)
        input_layout.addWidget(self.predict_button)

        # Create button for saving the output
        self.save_button = QPushButton("Save Output", self)
        self.save_button.clicked.connect(self.save_output)
        self.save_button.setEnabled(False)
        input_layout.addWidget(self.save_button)

        main_layout.addLayout(canvas_layout)
        main_layout.addSpacing(4)
        main_layout.addLayout(input_layout)

    def load_model_btn(self):
        try:
            # Train a new model
            if self.radio_button1.isChecked():
                model_ = Model()
                self.model = model_.model
                self.model_available = True
                self.browse_button.setEnabled(True)

            # Read an existing trained model
            elif self.radio_button2.isChecked():
                self.model = self.read_model()
                self.model_available = True
                self.browse_button.setEnabled(True)
            else:
                self.model_available = False
        except:
            self.model_available = False
            print(
                "[ERROR] Model not found. Please load an .h5 model.")

    def radio_button_clicked(self):
        self.load_model_button.setEnabled(True)

    def read_model(self):
        # Browse for an existing trained model
        file_dialog = QFileDialog()
        file_dialog.setNameFilter(
            ".h5 files (*.h5)")
        file_dialog.setWindowTitle("Select a Model")

        if file_dialog.exec():
            model_path = file_dialog.selectedFiles()[0]

        model = load_model(model_path)
        return model

    def browse_image(self):
        # Browse for input image to perform prediction
        file_dialog = QFileDialog()
        file_dialog.setNameFilter(
            "Image files (*.jpg *.jpeg *.png *.bmp *.tif *.tiff)")
        file_dialog.setWindowTitle("Select an Image")

        if file_dialog.exec():
            self.image_path = file_dialog.selectedFiles()[0]
            self.image = cv2.imread(self.image_path)
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

            # Display the input image on the input canvas
            if (len(self.image[0]) / len(self.image)) < (self.canvas_width / (self.canvas_height // 2)):
                input_pixmap = QPixmap(self.image_path).scaled(
                    int(len(self.image[0]) * ((self.canvas_height // 2) / len(self.image))), self.canvas_height // 2)
            else:
                input_pixmap = QPixmap(self.image_path).scaled(
                    self.canvas_width, int(len(self.image) * ((self.canvas_width) / len(self.image[0]))))
            self.input_label.setPixmap(input_pixmap)
            if self.model_available:
                self.predict_button.setEnabled(True)

    def predict_image(self):
        if self.model_available:
            # Clear previous predictions
            self.grid_images.clear()
            self.predictions.clear()

            # Get the number of rows and columns from the boxes
            rows = int(self.rows_input.text())
            columns = int(self.columns_input.text())

            # Divide the image into grid pieces
            self.grid_images = self.divide_image(self.image, rows, columns)

            # Predict and display
            self.predictions = self.predict(self.grid_images)
            self.display_results(self.predictions, rows, columns)

    def divide_image(self, image, rows, columns):
        # Divide the image into specified number of rows and columns
        grid_images = []
        image_height, image_width, _ = image.shape
        clip_width = image_width // max(1, columns)
        clip_height = image_height // max(1, rows)

        for i in range(rows):
            for j in range(columns):
                y = i * clip_height
                x = j * clip_width
                grid_image = image[y:y+clip_height, x:x+clip_width]
                grid_images.append(grid_image)

        return grid_images

    def predict(self, images):
        # Convert images to proper typea and size
        images = [cv2.resize(image, (100, 100)) for image in images]
        images = np.array(images)
        images = images.astype('float32') / 255.0

        # Perform prediction using the model
        predictions = self.model.predict(images)
        confidence_values = np.max(predictions, axis=1)

        # Process results
        predictions = np.argmax(predictions, axis=1)
        predictions = list(map(lambda p: self.ingredients[p], predictions))
        confidence_values = list(map(lambda c: round(c, 2), confidence_values))
        results = list(zip(predictions, confidence_values))

        return results

    def display_results(self, results, rows, columns):
        # Create the output
        fig = plt.figure(figsize=(12, 6))
        for i in range(rows * columns):
            result = results[i]
            plt.subplot(rows, columns, i + 1)
            plt.imshow(self.grid_images[i])
            plt.title(
                f"Prediction: {result[0]}\nConfidence: {result[1]:.2f}")
            plt.axis('off')
        plt.tight_layout()

        # Read output as image
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', facecolor="#f0f0f0")
        buffer.seek(0)
        image_data = buffer.getvalue()
        image_array = np.frombuffer(image_data, dtype=np.uint8)
        output_image = np.array(cv2.imdecode(
            image_array, cv2.IMREAD_UNCHANGED))

        image_height, image_width = output_image.shape[:2]

        # Display the output image on the output canvas
        if (image_width / image_height) < (self.canvas_width / (self.canvas_height // 2)):
            output_pixmap = QPixmap.fromImage(
                QImage(output_image.data, image_width, image_height, QImage.Format_RGB32))
            output_pixmap = output_pixmap.scaled(int(output_pixmap.width(
            ) * ((self.canvas_height // 2) / output_pixmap.height())), self.canvas_height // 2)
        else:
            output_pixmap = QPixmap.fromImage(
                QImage(output_image.data, image_width, image_height, QImage.Format_RGB32))
            output_pixmap = output_pixmap.scaled(self.canvas_width, int(
                output_pixmap.height() * ((self.canvas_width) / output_pixmap.width())))
        self.output_label.setPixmap(output_pixmap)

        self.save_button.setEnabled(True)

    def save_output(self):
        # Check if output directory exists, otherwise create
        target_dir = "./output_data"
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)

        # Get the current timestamp to use as the filename
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        input_image_name = os.path.splitext(
            os.path.basename(self.image_path))[0]
        filename = f"{target_dir}/{input_image_name}_predictions_{timestamp}.png"

        # Save the plot image
        plt.savefig(filename, format='png')


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.move(100, 100)
    window.show()
    sys.exit(app.exec_())
