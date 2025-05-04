# DECISION-TREE-IMPLEMENTATION
*COMPANY*: CODTECH IT SOLUTIONS
*NAME*: SAKSHI SAPKAL
*INTERN ID*: CT12WV77
*DOMAIN*: MACHINE LEARNING
*DURATION*: 12 WEEKS
*MENTOR*: NEELA SANTOSH

Decision Tree is one of the most widely used and intuitive algorithms in machine learning for both classification and regression problems. The primary objective of this project is to build, visualize, and analyze a Decision Tree classifier using the scikit-learn library on a real dataset. The entire workflow not only helps understand how the algorithm works but also highlights its interpretability and practical applications.
A Decision Tree is a supervised machine learning algorithm that uses a tree-like graph of decisions and their possible consequences. Each internal node of the tree represents a feature (or attribute), each branch represents a decision rule, and each leaf node represents an outcome (class label or target value). The model learns a hierarchy of "if-then-else" questions that lead to a prediction.
Decision Trees are popular because they are easy to interpret, handle both numerical and categorical data, and require little data preprocessing.

Dataset Used
In this implementation, we have used the Iris dataset, one of the classic and widely used datasets for pattern recognition. The dataset contains 150 samples of iris flowers, divided into three classes (Setosa, Versicolor, Virginica). The features include:

Sepal Length, Sepal Width, Petal Length, Petal Width
The target variable indicates the species of the flower.

Implementation Steps
Data Loading and Exploration
We first load the Iris dataset using scikit-learn.datasets. We inspect the dataset, including feature names, target names, data types, and class distribution, to understand its structure and confirm it is clean and balanced.

Train-Test Split
The dataset is split into training and testing subsets using an 80-20 ratio. This helps ensure the model is evaluated on unseen data, preventing overfitting and giving a fair estimate of its performance.

Model Training
We instantiate and train the DecisionTreeClassifier using the training data. By default, the classifier uses the Gini impurity criterion to select splits, but it also supports Entropy as an alternative.

Visualization
One of the biggest strengths of Decision Trees is their interpretability. We visualize the trained tree using plot_tree() from scikit-learn. The plot clearly shows which features are used for splitting, the thresholds, and the class distributions at each node. Such visualizations are invaluable for understanding model decisions and for communicating results to non-technical stakeholders.

Model Evaluation
We evaluate the model’s performance on the test set by calculating:

Accuracy score, Confusion matrix, Classification report (precision, recall, F1-score)
We also visualize the confusion matrix using a heatmap (via seaborn) to better interpret the performance across different classes.

Hyperparameter Tuning
To improve generalization and avoid overfitting, we fine-tune hyperparameters such as max_depth. Limiting the tree depth often simplifies the model without sacrificing much accuracy. After tuning, we retrain the model and compare the performance and tree complexity to the untuned version

OUTPUT: 
<!-- Uploading "TO1.png"... -->
