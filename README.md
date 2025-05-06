# Advanced-Breast-Cancer-Classification-Using-a-Hybrid-CNN-ViT-LSTM-Model
The Advanced Breast Cancer Classification project is a sophisticated deep learning initiative designed to classify breast ultrasound images into three categories: benign, malignant, and normal.
By integrating Convolutional Neural Networks (CNNs), Vision Transformers (ViTs), and Long Short-Term Memory (LSTM) networks, the project develops a hybrid model that leverages spatial, contextual, and sequential feature extraction to achieve robust classification performance. The system is trained on a publicly available breast ultrasound dataset and includes a prediction pipeline for real-time image classification. This project showcases expertise in medical image analysis, advanced neural network architectures, and healthcare-focused AI applications, with potential to assist in early breast cancer detection and diagnosis.

Objectives
Accurate Classification: Develop a deep learning model to classify breast ultrasound images into benign, malignant, or normal categories with high accuracy.
Hybrid Architecture: Combine CNNs for spatial feature extraction, ViTs for global context, and LSTMs for sequential dependencies to enhance model performance.
Robust Data Handling: Implement data augmentation and preprocessing to address variability in ultrasound images and improve model generalization.
Real-Time Prediction: Provide a pipeline for classifying new ultrasound images, enabling practical deployment in clinical settings.
Interpretability: Evaluate model performance through accuracy, loss curves, and validation metrics to ensure reliability and transparency.
Technical Approach
1. Data Preparation
Dataset: The project utilizes the "Breast Ultrasound Images Dataset" from Kaggle, containing 1,578 images across three classes: benign (891 images), malignant (421 images), and normal (266 images). The dataset is accessed via the kagglehub API and stored locally.
Preprocessing:
Images are resized to 224x224 pixels and normalized to a [0, 1] range.
Data augmentation techniques (rotation, width/height shift, shear, zoom, horizontal flip) are applied to increase dataset diversity and prevent overfitting.
Data Splitting: The dataset is split into 80% training (1,263 images) and 20% validation (315 images) sets using TensorFlow’s ImageDataGenerator with a validation split.
Data Loading: Images are loaded in batches of 32 using flow_from_directory, ensuring efficient memory usage and seamless integration with the model.
2. Model Architecture
The project employs a hybrid deep learning model combining CNNs, ViTs, and LSTMs to capture complementary features:

Convolutional Neural Network (CNN) Block:
Four convolutional layers (32, 64, 128, and 256 filters) with 3x3 kernels and ReLU activation extract spatial features.
Batch normalization stabilizes training, while max-pooling (2x2) reduces spatial dimensions.
Dropout (0.25 after convolutions, 0.5 after pooling) prevents overfitting.
Global average pooling produces a compact feature representation (256-dimensional).
Vision Transformer (ViT) Block:
The CNN output is reshaped into a sequence and processed by a multi-head attention layer (8 heads, 512 key dimensions) to capture global contextual relationships.
Layer normalization and a feed-forward network (512 units, ReLU) with dropout (0.5) enhance feature transformation.
A second dense layer and normalization restore the feature dimension, preserving information.
Long Short-Term Memory (LSTM) Block:
The ViT output is reshaped and fed into two LSTM layers (128 and 64 units) to model sequential dependencies, potentially capturing temporal patterns in feature sequences.
Dropout (0.5) is applied between LSTM layers to improve generalization.
Fully Connected Layers:
A dense layer with 512 units and ReLU activation integrates features.
Batch normalization and dropout (0.5) stabilize and regularize the output.
A softmax layer produces probabilities for the three classes (benign, malignant, normal).
Compilation: The model is compiled with the Adam optimizer, categorical cross-entropy loss, and accuracy as the evaluation metric. It contains 5.14 million trainable parameters.
3. Training and Evaluation
Training: The model was trained for 30 epochs with a batch size of 32, using the training generator (39 steps per epoch) and validation generator (9 steps per epoch).
Performance:
Validation accuracy reached 63.49%, with a validation loss of 0.8758.
Training accuracy peaked at 78.12% (Epoch 14), while validation accuracy peaked at 64.24% (Epoch 25).
The model showed signs of convergence, though validation accuracy fluctuated, suggesting potential overfitting or dataset limitations.
Visualization: Training and validation accuracy/loss curves were plotted using Matplotlib to assess model performance and identify training dynamics.
Evaluation: The model was evaluated on the validation set, providing loss and accuracy metrics. No confusion matrix or classification report was generated, but these could be added for deeper analysis.
4. Prediction Pipeline
Model Saving: The trained model is saved as advanced_breast_cancer_classification_model.h5 for reuse.
Real-Time Prediction: A predict_image function loads and preprocesses a single image, resizes it to 224x224, normalizes it, and predicts the class (benign, malignant, or normal) using the loaded model.
Deployment Readiness: The prediction pipeline is designed for integration into clinical tools or web applications, though no GUI was implemented in this codebase.
Key Features
Hybrid Architecture: Combines CNNs, ViTs, and LSTMs to capture spatial, contextual, and sequential features, making the model robust to complex ultrasound image patterns.
Data Augmentation: Extensive augmentation mitigates dataset imbalance and improves generalization.
Scalable Data Pipeline: TensorFlow’s ImageDataGenerator ensures efficient handling of large datasets.
Real-Time Prediction: The prediction function enables practical deployment for single-image classification.
Comprehensive Evaluation: Accuracy and loss curves provide insights into model performance, with room for additional metrics like precision and recall.
Technologies Used
Programming Languages: Python
Deep Learning Frameworks: TensorFlow, Keras
Data Handling: KaggleHub for dataset access, NumPy for array operations
Data Visualization: Matplotlib for plotting accuracy and loss curves
Libraries: TensorFlow’s ImageDataGenerator for preprocessing, Scikit-learn (implicitly for metrics)
Hardware: Trained on a system with GPU support (assumed, based on TensorFlow usage and training times)
Results
Model Performance: Achieved a validation accuracy of 63.49% and a validation loss of 0.8758 after 30 epochs, indicating moderate performance suitable for initial screening but requiring improvement for clinical use.
Training Dynamics: Training accuracy improved steadily (up to 78.12%), but validation accuracy plateaued around 64%, suggesting potential overfitting or dataset limitations (e.g., class imbalance or image variability).
Dataset Insights: The dataset’s imbalance (891 benign, 421 malignant, 266 normal) may have impacted performance, particularly for the underrepresented normal class.
Prediction Capability: The predict_image function successfully classifies new images, demonstrating practical applicability.
Challenges and Solutions
Challenge: Class imbalance in the dataset (benign > malignant > normal).
Solution: Applied data augmentation to increase dataset diversity and used categorical cross-entropy to handle multi-class classification.
Challenge: Overfitting due to complex model architecture (5.14M parameters).
Solution: Incorporated batch normalization, dropout (0.25–0.5), and global average pooling to regularize the model.
Challenge: Limited validation accuracy (63.49%).
Solution: Extended training to 30 epochs and used augmentation, though further tuning or a larger dataset could improve results.
Challenge: No GUI for end-user interaction.
Solution: Focused on a programmatic prediction pipeline, with potential for future GUI integration (e.g., using Tkinter or Flask).
Future Enhancements
Model Optimization: Experiment with hyperparameter tuning, learning rate schedules, or lighter architectures (e.g., MobileNet, EfficientNet) to improve accuracy and reduce overfitting.
Class Imbalance Handling: Implement techniques like class weighting or oversampling to better handle the imbalanced dataset.
Additional Metrics: Generate confusion matrices, precision, recall, and F1-scores to evaluate per-class performance.
Explainability: Add visualization techniques (e.g., Grad-CAM) to highlight regions influencing predictions, aiding clinical interpretability.
GUI Development: Build a user-friendly interface (e.g., Tkinter or web-based) for medical professionals to upload and classify images.
Clinical Validation: Collaborate with radiologists to test the model on real-world ultrasound data and refine it for clinical deployment.
Impact and Applications
This project has significant potential in medical diagnostics, particularly for early breast cancer detection. By classifying ultrasound images into benign, malignant, or normal categories, it can assist radiologists in triaging cases, reducing diagnostic workload, and improving patient outcomes. The hybrid CNN-ViT-LSTM architecture demonstrates innovation in combining advanced neural network techniques, making it adaptable to other medical imaging tasks. With further optimization, the model could be integrated into clinical workflows, contributing to AI-driven healthcare solutions.

Conclusion
The Advanced Breast Cancer Classification project exemplifies the application of state-of-the-art deep learning techniques to address a critical healthcare challenge. The hybrid CNN-ViT-LSTM model, trained on a breast ultrasound dataset, achieves moderate classification performance while demonstrating robustness through data augmentation and a scalable pipeline. Although validation accuracy (63.49%) suggests room for improvement, the project lays a strong foundation for medical image analysis and showcases transferable skills in AI, computer vision, and healthcare technology. With future enhancements, this system could become a valuable tool for early breast cancer detection, advancing both technical innovation and clinical impact.
