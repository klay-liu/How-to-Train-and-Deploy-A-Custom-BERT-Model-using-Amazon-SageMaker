Medium Blog: [How to Train and Deploy A Custom BERT Model using Amazon SageMaker](https://medium.com/@dwliuliu/how-to-train-and-deploy-a-custom-bert-model-using-amazon-sagemaker-c4f80705aa6)

### How to Train and Deploy A Custom BERT Model using Amazon SageMaker

Amazon Sagemaker is a machine learning platform for every developer and data scientist which enables developers to create, train, and deploy machine learning (ML) models in the cloud[ ](https://en.wikipedia.org/wiki/Amazon_SageMaker#cite_note-:6-2)and also enables developers to deploy ML models on embedded systems and edge-devices.Sagemaker offers built-in algorithms and scripting modes for training models.

## Content

This post will focus on building and deploying a custom BERT model in script mode based on my experience in my daily work.
You should write a model training script and a submission script to train the model before using Sagemaker’s script mode. The following sections make up the content of this post:

0. Dataset used in the post

1. Train a BERT model in Sagemaker
2. Deploy/Host a trained model in Sagemaker
3. How to create Sagemaker jobs with Submission Script

![img](https://cdn-images-1.medium.com/max/1000/1*CgBGM7SOSt6SuKlKJyy6KQ.png)																						Fig. 1 Content structure

## File Structure

```
./code                           --> to perform the inference with trained model
	__init__.py  
	inference.py  
	requirements.txt

./data:                          --> dataset used in this post
	samples.csv  
	test.csv  
	train.csv  
	Womens Clothing E-Commerce Reviews.csv

./scripts:                       --> train scripts for Sagemaker
	__init__.py  
	train.py
	
pytorch_bert-train-deploy.ipynb  --> Submission script

ReadMe.md

```

## How to use

Follow the steps in `pytorch_bert-train-deploy.ipynb`
