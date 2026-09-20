import wandb
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

wandb.login()
wandb.init(project="cca1-mlops-demo")

X, y = load_iris(return_X_y=True)
model = RandomForestClassifier(n_estimators=100, max_depth=5)
model.fit(X, y)
acc = model.score(X, y)

wandb.log({"accuracy": acc, "n_estimators": 100, "max_depth": 5})
wandb.finish()