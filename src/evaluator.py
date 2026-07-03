import mlflow
from sklearn.metrics import f1_score, classification_report

class ModelEvaluator:
    @staticmethod
    def evaluate(model, model_name, X_test, y_test): 
        y_pred = model.predict(X_test)
        
        f1 = f1_score(y_test, y_pred, average='macro')
        mlflow.log_metric("test_f1_macro", f1)
        
        print(f"Detailed report for {model_name}:")
        print(classification_report(y_test, y_pred))
        
        return f1