# import argparse
# import joblib
# import xgboost as xgb
# from sklearn.datasets import load_breast_cancer
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, classification_report

def main(args):
	# Load Breast Cancer dataset
	X, y = load_breast_cancer(return_X_y=True)
	# For target_names, load the dataset metadata
	data_meta = load_breast_cancer()
	# Split data into train/test
	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=args.test_size, random_state=42, stratify=y
	)

	# Define XGBoost classifier with hyperparameters
	model = xgb.XGBClassifier(
		n_estimators=args.n_estimators,
		max_depth=args.max_depth,
		learning_rate=args.learning_rate,
		subsample=args.subsample,
		colsample_bytree=args.colsample_bytree,
		eval_metric='logloss',
		random_state=42,
	)
	# Train the model
	model.fit(X_train, y_train)
	# Predict and evaluate
	preds = model.predict(X_test)
	acc = accuracy_score(y_test, preds)
	print(f"\nTest Accuracy: {acc:.4f}\n")
	print(classification_report(y_test, preds, target_names=data_meta.target_names))
	print(classification_report(y_test, preds, target_names=data.target_names))
	# Save the model
	joblib.dump(model, args.model_path)
	print(f"\n✅ Model saved to: {args.model_path}")

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Train an XGBoost model on the Breast Cancer dataset.")
	parser.add_argument("--n_estimators", type=int, default=1000, help="Number of boosting rounds.")
	parser.add_argument("--max_depth", type=int, default=6, help="Maximum depth of a tree.")
	parser.add_argument("--learning_rate", type=float, default=1.5, help="Boosting learning rate.")
	parser.add_argument("--subsample", type=float, default=1.0, help="Subsample ratio of the training instances.")
	parser.add_argument("--colsample_bytree", type=float, default=1.0, help="Subsample ratio of columns for each tree.")
	parser.add_argument("--test_size", type=float, default=0.2, help="Test set proportion.")
	parser.add_argument("--model_path", type=str, default="xgb_breast_cancer.joblib", help="Model save path.")
	args = parser.parse_args()
	main(args)