README: MNIST Classifier Deployment with Streamlit
Project Overview
This repository contains the implementation of an MNIST handwritten digit classifier, trained using TensorFlow and deployed with Streamlit for interactive use. Users can upload a digit image, and the model will classify it in real-time.
File Structure
mnist_classifier/
│── model/                      # Saved TensorFlow trained model
│   ├── mnist_cnn_model.h5      # CNN model file
│── app.py                       # Streamlit web application
│── requirements.txt             # Dependency list
│── README.md                    # Documentation file (this file)
│── sample_images/               # Example test images
│   ├── digit_0.png              # Sample handwritten '0'
│   ├── digit_1.png              # Sample handwritten '1'
│── screenshots/                 # Deployment images
│   ├── web_app_demo.png         # Screenshot of the deployed app
Installation & Setup
1. Clone the Repository
bash
git clone ADD CLONE LINK HERE
cd mnist_classifier
2. Install Dependencies
bash
pip install -r requirements.txt
3. Run Streamlit App
bash
streamlit run app.py
This will launch a web interface where users can upload digit images for classification.
Usage Guide
    • Upload a 28x28 grayscale handwritten digit image (.png, .jpg, .jpeg).
    • The model will predict the digit and display the result.
    • Sample images are available in the /sample_images/ folder for testing.
Deployment
    • Local Deployment: Run streamlit run app.py on your machine.
    • Cloud Deployment: Deploy on Streamlit Cloud or Render for public access.
Live Demo & Screenshots
🔗 Live Demo: ADD LINK HERE 📷 Preview: ADD LINK HERE
Contributors & License
Maintained by Dominik Kean. Licensed under MIT License.
