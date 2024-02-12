# Optimizing K-Nearest Neighbors for Web Phishing Detection through Genetic Algorithm-driven Feature Selection
  Groundbreaking approach aimed at enhancing the effectiveness of K-Nearest Neighbors (KNN) algorithm in identifying and mitigating web phishing threats. By integrating the power of Genetic Algorithms (GA) with KNN, this innovative method optimizes feature selection, thus significantly improving the algorithm's performance in distinguishing between legitimate and malicious web content.

## General Workflow
![Alt text](https://github.com/Skygers/Improving-KNN-for-Web-Phishing-with-Genetic-Algorithms/blob/9a887a71d76d481fca07da13fcf9620649a19ce4/images/Improving%20KNN%20for%20Web%20Phishing%20with%20Genetic%20Algorithms.png)

The workflow diagram contains the general process of improving KNN for web phishing detection using genetic algorithms. Here's a breakdown the **right side** of the workflow:
1. **Dataset**:
   - This dataset contains features extracted from websites to classify them as legitimate or phishing (malicious). Features include the use of IP addresses, length of URLs, presence of @ symbol, redirection, etc.
   - **Preview Dataset**![Preview Dataset](https://github.com/Skygers/Improving-KNN-for-Web-Phishing-with-Genetic-Algorithms/blob/9a887a71d76d481fca07da13fcf9620649a19ce4/images/dataset.png)

2. **Data Preprocessing**:
   - The process begins with data preprocessing, where the raw dataset undergoes various cleaning and transformation steps to prepare it for analysis. This may include handling missing values, encoding categorical variables, and scaling numerical features.

3. **Feature Selection**:
   - Next, feature selection is performed using genetic algorithms. This involves generating an initial population of feature subsets and iteratively evolving them through genetic operations such as selection, crossover, and mutation. The goal is to find the subset of features that maximizes the performance of the KNN classifier.
  
4. **Training KNN Classifier**:
   - The KNN classifier utilizes the entire feature set for training and prediction. It assigns each data point to the class of its nearest neighbors based on a predefined number of neighbors (K) and a distance metric. This straightforward approach considers all features, which can lead to increased computational complexity, especially in datasets with a large number of features. The performance of the KNN classifier depends heavily on the choice of K and the distance metric used. ![KNN Classifier](https://github.com/Skygers/Improving-KNN-for-Web-Phishing-with-Genetic-Algorithms/blob/e5820f73711e82b3c4fdf5ceb3a79ba9989c70f4/images/KNN_Working.gif)

     (https://padhaitime.com/Machine-Learning/K-Nearest-Neighbors)

5. **Training KNN Classifier with GA Feature Selection**
   - With the selected features, a KNN classifier is trained on the preprocessed dataset. The KNN algorithm assigns each data point to the class of its nearest neighbors based on a predefined number of neighbors (K) and a distance metric.

6. **Model Evaluation**:
   - The trained KNN classifier and The trained KNN classifier with GA feature selection is then evaluated using a separate validation dataset to assess its performance. Various evaluation metrics such as accuracy, precision, recall, and F1 score may be computed to measure the classifier's effectiveness in detecting web phishing attacks.

7. **Optimization Loop**:
   - The entire process may be iterated multiple times, with feedback from the evaluation phase used to fine-tune the parameters of the genetic algorithm and improve the performance of two model further.

## Comparative Evaluation:

**Performance Metrics**: Both models are evaluated using standard classification performance metrics such as accuracy, precision, recall, F1-score, and area under the ROC curve (AUC). These metrics provide a comprehensive assessment of the models' predictive capabilities, including their ability to correctly classify instances from different classes and their robustness to class imbalances.

**Computational Complexity**: The computational complexity of each model is assessed in terms of training time, prediction time, and memory usage. While the KNN classifier with GA feature selection may achieve better performance by reducing the feature space, it often requires more computational resources during the feature selection process.

**Generalization and Robustness**: The generalization ability and robustness of each model are evaluated using cross-validation techniques and by testing them on unseen or out-of-sample data. This analysis helps determine whether the feature selection process improves the model's ability to generalize to new data and whether it effectively reduces overfitting.

**Interpretability**: Finally, the interpretability of the models is considered, particularly in the context of feature selection. The KNN classifier with GA feature selection may produce a more interpretable model by highlighting the most relevant features for classification, aiding in understanding the underlying mechanisms driving the predictions.
