# ML-practice
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

# Random Forest 
- algorithm randomly selects a bootstrap sample to train on and a random sample of features to use at each split >> a more diverse set of trees, lessen tree correlation beyond bagged trees, increase predictive power.
### Parameters
**Features to improve the predictive power of the model**
- n_estimators : number of trees >> Higher number of trees give you better performance but makes your code slower ***(start with at least 10 times the number of features)***
- max_features: ***(Typical approach: p/3 for regression and square of p for classification)***
- Tree complexity ( node size, max depth, required node for split): ***Node size | min_sample_leaf: 1 for classification and 5 for regression. If data is noisy and higher number of feature in each tree >> increase node size (decreasing tree depth and complexity) will improve performance***

**attributes which have a direct impact on model training speed**
- n_jobs : -1 >> faster
- oob_score: random forest cross validation method

# Regularize Regression: to prevents the model from overfitting by adding extra information to it
- Use when there is a larger number of features
- Multicollinearity in the data
### Ridge Regression (L2 regularization): 
- It shrinks the parameters (reduce the coefficient of inputs))  >> mostly used to prevent multicollinearity, never sets the value of coefficients to absolute zero
- a ridge model is good if you believe there is a need to retain all features
### Lasso Regression (L1 regularization)
- Generally used when we have more number of features, because it automatically does feature selection. Lasso regression tends to make coefficients to absolute zero (when two strongly correlated features are pushed towards zero, one may be pushed fully to zero while the other remains in the model)
