import boto3
import sagemaker
from sagemaker.sklearn.model import SKLearnModel


BUCKET = "uas-md-shafi-2671"
MODEL_S3_KEY = "credit-score/model.tar.gz"
ENDPOINT_NAME = "credit-score-endpoint-5"


REGION = "us-east-1"
INSTANCE_TYPE = "ml.m5.large"
FRAMEWORK_VERSION = "1.2-1"

def get_lab_role_arn() -> str:
    """Mengambil IAM Role otomatis dari environment AWS Lab."""
    iam = boto3.client("iam")
    return iam.get_role(RoleName="LabRole")["Role"]["Arn"]

def main() -> None:
    boto3.setup_default_session(region_name=REGION)
    sm_session = sagemaker.Session()
    role_arn = get_lab_role_arn()
    model_s3_uri = f"s3://{BUCKET}/{MODEL_S3_KEY}"

    print(f"Role:      {role_arn}")
    print(f"Model URI: {model_s3_uri}")
    print(f"Endpoint:  {ENDPOINT_NAME}")

    model = SKLearnModel(
        model_data=model_s3_uri,
        role=role_arn,
        entry_point="inference.py",
        source_dir="src", 
        framework_version=FRAMEWORK_VERSION,
        sagemaker_session=sm_session,
    )

    print("\nDeploying endpoint (5-8 minutes)...")
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type=INSTANCE_TYPE,
        endpoint_name=ENDPOINT_NAME,
    )

    sample = {
        "instances": [
            {
                "Age": 30,
                "Annual_Income": 50000.0,
                "Monthly_Inhand_Salary": 4000.0,
                "Num_Bank_Accounts": 2,
                "Num_Credit_Card": 2,
                "Interest_Rate": 12.0,
                "Num_of_Loan": 1,
                "Delay_from_due_date": 10,
                "Num_of_Delayed_Payment": 3,
                "Changed_Credit_Limit": 10.0,
                "Num_Credit_Inquiries": 2,
                "Outstanding_Debt": 1500.0,
                "Credit_Utilization_Ratio": 35.0,
                "Credit_History_Age": 150,
                "Total_EMI_per_month": 100.0,
                "Amount_invested_monthly": 100.0,
                "Monthly_Balance": 500.0,
                "Occupation": "Engineer",
                "Credit_Mix": "Standard",
                "Payment_Behaviour": "High_spent_Small_value_payments",
                "Payment_of_Min_Amount": "Yes",
                "Type_of_Loan": ["Auto Loan", "Personal Loan"]
            }
        ]
    }

    runtime = boto3.client("sagemaker-runtime", region_name=REGION)
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Accept="application/json",
        Body=str(sample).replace("'", '"'),
    )
    
    print("\nSmoke test response:")
    print(response["Body"].read().decode("utf-8"))

    print(
        f"\nEndpoint '{ENDPOINT_NAME}' is live in {REGION}.\n"
        f"Delete it before lab teardown: predictor.delete_endpoint()"
    )

if __name__ == "__main__":
    main()