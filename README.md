# Loan-Default-Prediction

This dataset has been taken from Coursera's Loan Default Prediction Challenge https://www.coursera.org/projects/data-science-coding-challenge-loan-default-prediction.

https://www.kaggle.com/datasets/yasserh/loan-default-dataset


# AWS-CICD-Deployment-with-Github-Actions

# Step1: Build image locally. Tool: Docker
- Open terminal in the project and run: docker build -t loan_default_prediction:latest .
- verify the image was built: in the terminal, run: docker images.
- Run the container locally: docker run -p 8000:8000 loan_default_prediction:latest


## 1. Login to AWS console.

## 2. Create IAM user for deployment.
#with specific access

1. EC2 access : It is virtual machine

2. ECR: Elastic Container registry to save your docker image in aws


#Description: About the deployment

1. Build docker image of the source code

2. Push your docker image to ECR

3. Launch Your EC2 

4. Pull Your image from ECR in EC2

5. Lauch your docker image in EC2

#Policy:

1. AmazonEC2ContainerRegistryFullAccess

2. AmazonEC2FullAccess


## 3. Create ECR repo to store/save docker image
- Save the URI: 566373416292.dkr.ecr.us-east-1.amazonaws.com/text-s


# 4. Create EC2 machine (Ubuntu)

# 5. Open EC2 and Install docker in EC2 Machine:
#optinal

sudo apt-get update -y

sudo apt-get upgrade

#required

curl -fsSL https://get.docker.com -o get-docker.sh

sudo sh get-docker.sh

sudo usermod -aG docker ubuntu

newgrp docker

# 6. Configure EC2 as self-hosted runner:
setting>actions>runner>new self hosted runner> choose os> then run command one by one

# 7. Setup github secrets:

AWS_ACCESS_KEY_ID=

AWS_SECRET_ACCESS_KEY=

AWS_REGION = us-east-1

AWS_ECR_LOGIN_URI = demo>>  566373416292.dkr.ecr.ap-south-1.amazonaws.com

ECR_REPOSITORY_NAME = simple-app