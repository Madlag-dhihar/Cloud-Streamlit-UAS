import mlflow
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score  
from config import MODEL_DIR

class ModelTrainer:
    def __init__(self):
        self.models = {
            "RandomForest": {
                "estimator": RandomForestClassifier(random_state=42),
                "params": {
                    'n_estimators': [100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                }
            },
            "LogisticRegression": {
                "estimator": LogisticRegression(max_iter=1000, random_state=42),
                "params": {
                    'C': [0.01, 0.1, 1, 10],
                    'solver': ['lbfgs', 'liblinear']
                }
            },
            "XGBoost": {
                "estimator": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
                "params": {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1],
                    'max_depth': [3, 5]
                }
            }
        }
        
        self.best_model = None
        self.best_model_name = ""
        self.best_score = 0.0
        self.all_results = [] 

    
    def train_and_tune(self, X_train, y_train, X_test, y_test):
        for model_name, config in self.models.items():
            
            with mlflow.start_run(run_name=f"Tuning_{model_name}", nested=True):
                grid_search = GridSearchCV(
                    estimator=config["estimator"],
                    param_grid=config["params"],
                    cv=3,
                    scoring='f1_macro',
                    n_jobs=-1,
                    verbose=0 
                )
                
                grid_search.fit(X_train, y_train)

                best_current_model = grid_search.best_estimator_
                cv_f1_score = grid_search.best_score_
                best_current_params = grid_search.best_params_

                y_pred = best_current_model.predict(X_test)
                test_acc = accuracy_score(y_test, y_pred)
                test_f1 = f1_score(y_test, y_pred, average='macro')

                
                self.all_results.append({
                    "Model": model_name,
                    "Train F1": cv_f1_score,
                    "Test Acc": test_acc,
                    "Test F1": test_f1
                })

                mlflow.log_params(best_current_params)
                mlflow.log_metric("cv_f1_macro", cv_f1_score)
                mlflow.sklearn.log_model(
                    sk_model=best_current_model, 
                    artifact_path=f"{model_name}_model",
                    serialization_format="cloudpickle"
                )

                if cv_f1_score > self.best_score:
                    self.best_score = cv_f1_score
                    self.best_model = best_current_model
                    self.best_model_name = model_name

        
        print("\nModel\t\t\tTrain F1   Test Acc   Test F1")
        print("-" * 65)
        for res in self.all_results:
            
            print(f"{res['Model']:<24}{res['Train F1']:<11.4f}{res['Test Acc']:<11.4f}{res['Test F1']:.4f}")

        print("\nWinner:", self.best_model_name, "\n")
        
        joblib.dump(self.best_model, f"{MODEL_DIR}/best_model.pkl", compress=3)
        return self.best_model, self.best_model_name