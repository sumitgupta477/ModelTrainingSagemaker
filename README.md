# ModelTrainingSagemaker
Continously Train Model With Sagemaker

1. Create an AWS account
2. Download and install AWS cli, and configure with your iam admin service credentials and us-east-1 region
3. Download and install terraform
4. Run the terraform code
5. Make sure to have your s3 bucket(over here it is called mlops-llm-bucket-12345) populated with user data in mlops-llm-bucket-12345/input. Right now only txt files trigger auto model training.
6. Make sure to have your s3 bucket(over here it is called mlops-llm-bucket-12345) populated with training python code(packaged in tar.gz) in mlops-llm-bucket-12345/code.
7. The llm model will be stored in <s3>/llm-model/output/<training job name>/output/<model>
8. We have used hugging face llm for training and inference, feel free to change it. You can put your own inference script in mlops-llm-bucket-12345/code.
9. The code doesnt have a frontend, please feel free to use curl instead for example: curl -X POST "https://<api-gw-link>.execute-api.us-east-1.amazonaws.com/chat" -H "Content-Type: application/json" -d "{\"prompt\":\"Hello?\"}"
In the above, feel free to replace <api-gw-link> with your api gateway link.
