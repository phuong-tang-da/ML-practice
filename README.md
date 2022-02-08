# ML-practice
# Main kinds of learning problems
- Classification problem: predicting among two or more discrete classes
- Regression problem: predicting a continuous value

# Data spliting
- Training error (Error on the training data) & Generalization error (Error on the entire distribution  of data) >> data spliting is a way to approximate generalization error
- Train data: used for train model
- Validation data: hyperparameter tuning and model assessment
- Test data: use it once to evaluate the performance of the best performing model
- "Deployment" data: if our model does well on the validation and test data, we hope it will do well on deployment data.

# Cross-validation (using cross_val_score & cross_validate)
- It creates cv folds on the data.
- In each fold, it fits the model on the training portion and scores on the validation portion.
- The output is a list of validation scores in each fold.

# Underfitting, overfitting
- Underfitting: The model is so simple that it doesn't even capture the patterns in the training data
- Overfitting: Complex model, high variance, low bias >> Method to reduce overfit: Cross-validation,  train with more data, feature selection, regularization, ensembling (combining prediction from multiple model)

# Ensemble methods: improve performance by combining several of models 
### Bagging (“bootstrap aggregating”)- reduce variance: 
Bootstrapping: generating bootstrap samples from an initial dataset by randomly drawing with replacement. >> Fit several independent models on bootstrap sample and “average” their predictions in order to obtain a model with a lower variance. Ex: random forest (start with deep tree)
### Boosting- reduce bias: model is trained sequentially >> each model in the sequence is fitted giving more importance to observations in the dataset that were badly handled by the previous models in the sequence. Ex: adaboost and gradient boosting
-	AdaBoost (Adaptive Boosting): predicting original data set and gives equal weight to each observation > finding those examples in the training dataset that were misclassified, and adding more weight to those examples in the next iteration >> continue until a limit is reached 
-	Gradient Boosting: Each new model gradually minimizes the loss function  of the whole system using Gradient Descent method
-	Extreme Gradient Boosting (XGBoost): Only deal with numeric features (encoding needed for categorical features), XGBoost is capable of handling missing values internally
-	Cat Boost
-	LightGBM
### Stacking-different learning algorithms are combined: different weak learners are fitted independently from each others and a meta-model is trained on top of that to predict outputs based on the outputs returned by the base models

# Regularize Regression: to prevents the model from overfitting by adding extra information to it
- Use when there is a larger number of features
- Multicollinearity in the data
### Ridge Regression (L2 regularization): 
- It shrinks the parameters (reduce the coefficient of inputs))  >> mostly used to prevent multicollinearity, never sets the value of coefficients to absolute zero
- a ridge model is good if you believe there is a need to retain all features
### Lasso Regression (L1 regularization)
- Generally used when we have more number of features, because it automatically does feature selection. Lasso regression tends to make coefficients to absolute zero (when two strongly correlated features are pushed towards zero, one may be pushed fully to zero while the other remains in the model)

--------------------------------------------------------------------------------------------------
# Decision Tree
- Gini is intended for continuous attributes, Entropy is for attributes that occur in classes
- Gini is to minimize misclassification
- Entropy is for exploratory analysis, slower to compute
### Pros
- Decision trees are easy to interpret and visualize.
- It can easily capture Non-linear patterns.
- It requires fewer data preprocessing from the user, for example, there is no need to normalize columns.
- It can be used for feature engineering such as predicting missing values, suitable for variable selection.
- The decision tree has no assumptions about distribution because of the non-parametric nature of the algorithm
- Gives feature importance by default. >> feature selection. Feature importance in random forests is the ratio of how many times the feature was used in all trees that were created vs. number of trees in the forest.
### Cons
- Sensitive to noisy data. It can overfit noisy data.pruning techniques (e.g., minimum number of samples required at a leaf node, maximum depth of tree) is needed to avoid the problem
- The small variation(or variance) in data can result in the different decision tree. This can be reduced by bagging and boosting algorithms.
- Decision trees are biased with imbalance dataset, so it is recommended that balance out the dataset before creating the decision tree.
- They suffer from an inherent instability, since due to their hierarchical nature, the effect of an error in the top splits propagate down to all of the splits below.
- Is a "greedy" algorithm. Each node is only locally optimal which cannot gaurantee globally optimal tree

--------------------------------------------------------------------------------------------------
# Random Forest 
- algorithm randomly selects a bootstrap sample to train on and a random sample of features to use at each split >> a more diverse set of trees, lessen tree correlation beyond bagged trees, increase predictive power.
### HyperParameters
**Features to improve the predictive power of the model**
- n_estimators : number of trees >> Higher number of trees give you better performance but makes your code slower ***(start with at least 10 times the number of features)***
- max_features: ***(Typical approach: p/3 for regression and square of p for classification)***
- Tree complexity ( node size, max depth, required node for split): ***Node size | min_sample_leaf: 1 for classification and 5 for regression. If data is noisy and higher number of feature in each tree >> increase node size (decreasing tree depth and complexity) will improve performance***

**attributes which have a direct impact on model training speed**
- n_jobs : -1 >> faster
- oob_score: random forest cross validation method

--------------------------------------------------------------------------------------------------
# ADABoost (Adaptive Boosting)
- predicting original data set and gives equal weight to each observation > finding those examples in the training dataset that were misclassified, and adding more weight to those examples in the next iteration >> continue until a limit is reached 
### HyperParameters
- Number of Trees (n_estimators): high number of tree >> overfitting 
- Learning rate
- Weak learner

--------------------------------------------------------------------------------------------------
# XGBoost
###Common hyperparameter
- booster: gbtree (default),gblinear or dart
- learning_rate: step size shrinkage used to prevent overfitting. Range is [0,1].alues range from 0–1 with typical values between 0.001–0.3
- max_depth: Controls the depth of the individual trees. Typical values range from a depth of 3–8 but it is not uncommon to see a tree depth of 1 
- subsample: percentage of samples used per tree. Low value can lead to underfitting
- colsample_bytree: percentage of features used per tree. High value can lead to overfitting.
- n_estimators: number of trees you want to build. GBMs often require many trees (it is not uncommon to have many thousands of trees) but since they can easily overfit
- objective: determines the loss function to be used like

### regularization parameters to penalize models as they become more complex and reduce them to simple (parsimonious) model
- gamma: controls whether a given node will split based on the expected reduction in loss after the split. A higher value leads to fewer splits. Supported only for tree-based learners.
- alpha: L1 regularization on leaf weights. A large value leads to more regularization.
- gambda: L2 regularization on leaf weights and is smoother than L1 regularization.
--------------------------------------------------------------------------------------------------
# Save and Load Machine Learning Models in Python with scikit-learn
- https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/

--------------------------------------------------------------------------------------------------
# Naive Bayes Classifiers
- a collection of classification algorithms
- Assumption: each feature makes an independent & equal contribution to the outcome
- P(A|B)= P(B|A) * P(A) / P(B) : B is evidence, A in priori, A|B is posteriori of B
- P(y|X) = P(Yyes) * P(x1|Yyes)* P(x..|Yyes) * P (xn|Yyes) + P(Yno) * P(x1|Yno)* P(x..|Yno) * P (xn|Yno)
### Popular Naive Bayes classifiers: 
- Gaussian Naive Bayes: supports continuous values and has an assumption that each class is normally distributed. 
- Multinomial Naive Bayes: frequency features (document classification)
- Bernoulli Naive Bayes: features are independent binary (0,1) (document classification, text classification with ‘bag of words’ model)
### Pros:
- easy & fast
- well perform for categorical input variables
### Cons:
- probability outputs from predict_proba are not to be taken too seriously
- impossible assumption
### Application:
- Real time 
- Multiclass prediction
- Text classification: Spam, Sentiment Analysis

--------------------------------------------------------------------------------------------------
# Support Vector Machines 
- widely used in classification objectives
- Suitable for small, medium but comlicated dataset
- SVM finds an optimal hyperplane (decision plane) in n-dimensional space (n- number of features) to separate new data points -> Optimal mean: maximizing the distances between nearest data point (either class) and hyper-plane 
- A margin is a gap between the two lines on the closest class points
- for non-linear and inseparable planes, SVM uses a kernel trick to transform the input space to a higher dimensional space
### Tuning Hyperparameters
- Kernel: transform the given dataset input data into the required form: linear, polynomial, and radial basis function (RBF)
- c Regularization: C parameter adds a penalty for each misclassified data point to control the trade-off between decision boundary and misclassification term
-- small c -> low penalty -> a large margin (high misclassifications)
-- large c -> high pernalty -> small margin (overfitting)
- Gamma (RBF kernel): controls the distance of influence of a single training point 
-- low gamma -> large radius -> all points are group together
-- high gamma -> small radius ->point need to be very close (overfit(
### Pros:
- effective in high dimensional spaces
- effective in cases where the number of dimensions is greater than the number of samples.
- uses a subset of training points in the decision function -> memory efficient
- robust to outliers
### Cons:
- doesn’t perform well when we have large data set
- doesn’t perform very well, when the data set has more noise
- doesn’t directly provide probability estimates
