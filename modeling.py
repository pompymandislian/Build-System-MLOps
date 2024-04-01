import pandas as pd
import numpy as np
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score, roc_curve
from sklearn.preprocessing import StandardScaler
import os
import pickle
import matplotlib.pyplot as plt
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from mlflow.models import infer_signature

# Start the MLflow Tracking Server
# Run this command in your terminal:
# mlflow ui

# Set the correct tracking server URI
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")

# Create a new MLflow Experiment
mlflow.set_experiment("puchased-prediction-model")

# Set random seed for reproducibility
random_seed = 42
np.random.seed(random_seed)

# Generate dummy data
data = {
    'transaction': np.random.randint(1_000_000, 100_000_000, size=100),
    'age': np.random.randint(18, 65, size=100),
    'tenure': np.random.randint(0, 10, size=100),
    'num_pages_visited': np.random.randint(1, 20, size=100),
    'has_credit_card': np.random.choice([True, False], size=100),
    'items_in_cart': np.random.randint(0, 10, size=100),
    'purchase': np.random.choice([True, False], size=100)
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Split data into features and target
X = df.drop('purchase', axis=1)
y = df['purchase']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)

# Scale numeric features
scaler = StandardScaler()
numeric_features = ['num_pages_visited', 'items_in_cart', 'transaction']
X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test[numeric_features] = scaler.transform(X_test[numeric_features])

# Define the objective function to minimize (e.g., based on cross-validation performance)
def objective(params):
    model = LogisticRegression(random_state=random_seed, **params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = roc_auc_score(y_test, y_pred)
    return {'loss': -score, 'status': STATUS_OK}

# Define the search space
space = {
    "C": hp.loguniform("C", np.log(0.01), np.log(100)),
    "solver": hp.choice("solver", ["liblinear", "lbfgs", "saga"]),
    # Add other hyperparameters as needed
}

# Perform hyperparameter optimization
trials = Trials()
best_params = fmin(objective, space, algo=tpe.suggest, max_evals=100, trials=trials)

# Convert the index of the chosen solver back to the solver string
solver_choices = ["liblinear", "lbfgs", "saga"]
best_params["solver"] = solver_choices[best_params["solver"]]

# Log the best hyperparameters
mlflow.log_param("C", best_params["C"])
mlflow.log_param("solver", best_params["solver"])

# Initialize and train logistic regression model with the best hyperparameters
best_model = LogisticRegression(random_state=random_seed, **best_params)
best_model.fit(X_train, y_train)

# Predict on the testing set using the best model
y_pred = best_model.predict(X_test)

# Calculate metrics
roc_auc = roc_auc_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)

print("ROC-AUC:", roc_auc)
print("F1 Score:", f1)
print("Recall:", recall)
print("Precision:", precision)

# Get predicted probabilities for the positive class
y_scores = best_model.predict_proba(X_test)[:, 1]

# Compute ROC curve
fpr, tpr, _ = roc_curve(y_test, y_scores)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")

# Save ROC curve plot as a file
roc_plot_path = "roc_curve_plot.png"
plt.savefig(roc_plot_path)
plt.close()

# Infer the model signature
signature = infer_signature(X_train, best_model.predict(X_train))

# Log the metrics and ROC curve plot as MLflow artifacts
with mlflow.start_run():
    # Log the model
    model_info = mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="learn-model",
        signature=signature,
        input_example=X_train,
        registered_model_name="tracking-model",
    )

    # Log the metrics
    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("precision", precision)

    # Log the ROC curve plot as an artifact
    mlflow.log_artifact(roc_plot_path, artifact_path="roc_curve_plots")

    # Set a tag that we can use to remind ourselves what this run was for
    mlflow.set_tag("Training Info", "Learning Machine Learning ops")

# Remove temporary ROC curve plot file
os.remove(roc_plot_path)

# Save the trained model to a file
model_filename = "logistic_regression_model.pkl"
with open(model_filename, 'wb') as file:
    pickle.dump(best_model, file)
