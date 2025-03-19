# **Machine Learning Coursework: Forecasting, Clustering & Classification**

## **Overview**  
This project explores various machine learning techniques, including **forecasting ecological data, clustering high-dimensional data, and training classification models for medical image analysis**. The coursework is divided into two parts.

## **Part 1: Forecasting Haggis Population**  

This section applies **regression models** to predict population trends using **Gaussian basis functions**, **sinusoidal basis functions**, and **ridge regression with gradient descent**.

### **Key Tasks**
1. **Data Splitting Strategy**  
   - Designed an approach to **split the dataset** into **training, validation, and test sets** to optimize model performance.  

2. **Gaussian Basis Implementation**  
   - Developed a function to generate **Gaussian basis functions**, spacing them **uniformly** across the data range.  

3. **Ordinary Least Squares Regression**  
   - Implemented **gradient-based optimization** for **ridge regression (L2 regularization)**.  

4. **Combined Gaussian & Sinusoidal Basis Functions**  
   - Designed a **hybrid basis** capturing **annual oscillations** in population data.  

5. **Hyperparameter Selection & Model Evaluation**  
   - Used **validation data** to tune:
     - Number of basis functions  
     - Width of Gaussian basis functions  
     - Regularization term  
   - Generated **visualizations** comparing training data with true population values.  

## **Part 2: Clustering & Classification**  

### **Clustering & Dimensionality Reduction**
- Used the **UCI Human Activity Recognition Dataset** with **561 features**.  
- Applied **K-Means clustering** to group activities.  
- Evaluated clustering quality with metrics like **Silhouette Score**.  
- Used **PCA** for dimensionality reduction and visualized clusters in a **2D space**.  

### **Classification of Medical Images**
- Worked with the **BloodMNIST** dataset.  
- Trained **four different classifiers**, including logistic regression and neural networks.  
- Optimized **hyperparameters** and compared model performance.  
- Generated plots comparing **accuracy and loss** across models.  

## **Technologies Used**
- **Python (NumPy, Pandas, Scikit-learn, PyTorch, Matplotlib, Seaborn)**
- **Machine Learning Techniques: Regression, Clustering, Neural Networks**
- **Dimensionality Reduction: PCA**
- **Evaluation Metrics: Accuracy, Silhouette Score, Precision-Recall, Loss Curves**