# **Emotion Recognition from Bodily Movements**

This project was part of the **"Machine Learning for Emotion Recognition"** seminar organized by researchers at **DFKI (German Research Center for Artificial Intelligence)** in Summer Semester 2024. Below are the links to the final presentation and report:

- [Seminar Information - Machine Learning for Emotion Recognition SS2024](https://affective.dfki.de/teaching-2/seminar-machine-learning-for-emotion-recognition-ss2024/)
- [Final Presentation (PPT)](https://docs.google.com/presentation/d/1g-cGo7w9VEth5oDWDuU0CqsSvGvHQOfi)
- [Final Report (PDF)](https://drive.google.com/file/d/1oheb3TqAUacLsPymlwoGMEYgtSnhOU0v)

## **Project Overview**

Emotion recognition from bodily behavior plays a critical role in diagnosing mental disorders and enhancing human-computer interaction. This repository explores a machine learning model designed to recognize emotions based on **body language** in video data. The project focuses on extracting and analyzing key emotional features like **Activation**, **Valence**, **Power**, and **Anticipation** using various pre-trained action recognition models. 

### **Abstract**
Emotion recognition from bodily behavior is critical for understanding human interactions and diagnosing mental disorders. This project explores a machine learning model designed to recognize emotions based on **body language** in video data. Using the **MPIIEmo dataset**, we extract key emotional features—**Activation**, **Valence**, **Power**, and **Anticipation**—with the help of pre-trained action recognition models such as **SlowFast**, **I3D**, and **TSN**. Among these, **I3D** performed the best in capturing both spatial and temporal dimensions. The extracted features are then classified using an **SVM classifier**.

The model is evaluated using various generalization strategies, including **actor pair generalization** and **mixed views**, to assess its robustness across different scenarios and viewpoints.

## **Dataset**

The model utilizes the **MPIIEmo dataset**, which provides video data capturing various emotions through body language. Key attributes in the dataset include:
- **Activation**
- **Valence**
- **Power**
- **Anticipation**

The dataset is pre-processed to extract these features from the video sequences, which are then used for training and testing the model.

## **Methodology**

### **1. Feature Extraction**
For feature extraction, we leveraged **pre-trained action recognition models** for transfer learning, which allowed us to capture complex motion patterns without training from scratch. The models used include:
- **SlowFast**
- **TSN (Temporal Segment Networks)**
- **I3D (Inflated 3D Convolutional Network)**

Among these models, **I3D** performed better in capturing both the **spatial** and **temporal** dimensions of body movements in video data, making it the primary model for extracting features in this project. These extracted feature vectors represent the critical emotional components and are used for further classification.

### **2. Classification**
Once the features are extracted, an **SVM (Support Vector Machine)** classifier is employed to categorize the emotions based on the extracted vectors. The emotion labels include **Activation, Valence, Power,** and **Anticipation**.

### **3. Evaluation Strategies**
To assess the generalization capability of the model, three distinct evaluation strategies are implemented:

- **Generalization Across Actor Pairs**: The model is trained and tested using distinct actor pairs to evaluate how well it can generalize across different individuals.
  
- **Generalization Across Views**: The model is evaluated based on its ability to handle videos captured from different perspectives (camera angles), ensuring robustness to viewpoint changes.

- **Mixed Views**: A combined approach where the model is trained with mixed actor pairs and camera angles, simulating real-world variability to further test generalization.

## **Evaluation**

The model’s performance is measured using various metrics to evaluate its accuracy in classifying emotions. The evaluations consider the model’s robustness across different actors and camera views, focusing on how well it generalizes to unseen data.

### **Results**

- The **generalization across actor pairs** showed that the model performed effectively, achieving high classification accuracy when tested on new actor pairs.
- The **mixed views approach** provided a deeper understanding of how the model handles varying scenarios, showing potential for real-world applications where body movements are captured from diverse perspectives.

## **Technologies Used**

- **Python**: Core programming language for building the model.
- **TensorFlow/Keras**: For implementing the I3D convolutional network and utilizing pre-trained models for transfer learning.
- **Scikit-learn**: Used for implementing the SVM classifier and evaluation metrics.
- **OpenCV**: For video processing and handling input data.
- **Pandas/Numpy**: For data manipulation and preprocessing.
- **Matplotlib/Seaborn**: For visualizing the results and analysis.

## **Installation and Setup**

1. Clone this repository:

   ```bash
   git clone https://github.com/deepalipawade/Emotion-Recognition-From-Bodily-Movements.git
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Download the **MPIIEmo** dataset from the official source and place it in the appropriate folder.

4. Run the notebooks or scripts for training and evaluation:

   ```bash
   python train_model.py
   ```

## **Conclusion**

This project demonstrates the potential of using body movements for emotion recognition. By leveraging **pre-trained action recognition models** and capturing the spatial and temporal dynamics of videos, the model can accurately classify emotions. With robust evaluation strategies, this project sets a foundation for future research and development in areas like mental health diagnostics and human-computer interaction.

## **Future Work**

- Further fine-tuning of the feature extraction process.
- Exploration of other classification models for improved accuracy.
- Expanding the dataset to include more diverse body movements and emotions.
