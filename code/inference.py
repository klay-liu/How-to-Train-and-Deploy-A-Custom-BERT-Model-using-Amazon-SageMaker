import os
import pandas as pd
import torch
import numpy as np
import random
from transformers import BertTokenizer, BertModel, AdamW
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import TensorDataset
import sys
import logging
import argparse
import torch.nn.functional as F
from io import StringIO

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

class BertClassifier(nn.Module):
    def __init__(self, n_classes=2):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.drop = nn.Dropout(p=0.5)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
        
    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict = False
        )
        output = self.drop(pooled_output)
        return self.out(output)
    

def model_fn(model_dir):
    model = BertClassifier()
    model_path = os.path.join(model_dir, "model", 'model.bin')
    
    if (device == torch.device("cpu")) or (device=="cpu"):
                model.load_state_dict(
            torch.load(model_path, map_location=torch.device('cpu')))
    else:
        model.load_state_dict(
            torch.load(model_path))

    # with open(os.path.join(model_dir, "model.bin"), "rb") as f:
    #     model.load_state_dict(torch.load(f))
    logger.info('Successfully loaded the model')
    return model.to(device)

def input_fn(request_body, request_content_type):
    """
    Deserialize and prepare the prediction input
    """
    logger.info(f'request_body: {request_body}')
    
    if request_content_type == "application/json":
        sentences = [json.loads(request_body)]
    elif request_content_type == "text/csv":
        # We have a single column with the text.
        sentences = list(pd.read_csv(StringIO(request_body)).values[:, 0].astype(str))
    else:
        sentences = [request_body]
    logger.info(f'sentences: {sentences}')
    return sentences

def predict_fn(input_data, model):
    """
    Apply model to the incoming request
    """   
#     logger.info(f'input_data: {input_data}')
    
    input_ids = []
    attention_masks = []
    logger.info(f'input_data: {input_data}')
    for sent in input_data:
        encoded_input = tokenizer(
            sent,
            add_special_tokens=True,
            max_length=200,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt', 
            )
        
        # Add the encoded sentence to the list.    
        input_ids.append(encoded_input['input_ids'])
        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_input['attention_mask'])
        
    logger.info(f'input_ids: {input_ids}')
    logger.info(f'attention_masks: {attention_masks}')
    
    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    
    input_dataset = TensorDataset(input_ids, attention_masks)
    dataloader = DataLoader(
        input_dataset, batch_size=64
    )
    
    predictions = []
    prediction_probs = []
    
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            outputs = model(input_ids=batch[0].to(device), attention_mask=batch[1].to(device))
            logger.info(f'outputs: {outputs}')
            _, preds = torch.max(outputs, dim=1)
            y_probability=F.softmax(outputs, dim=1)
            probs = torch.max(y_probability, dim=1)[0]
            predictions.extend(preds)
            prediction_probs.extend(probs)

    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    
    return predictions, prediction_probs

def output_fn(prediction, response_content_type):
    """
    Serialize and prepare the prediction output
    """
    pred, score = prediction
    print("In output_fn")
    logging.info("Info: output_fn")

    if response_content_type == "application/json":
        response = pd.DataFrame({'pred': pred, 'score': score}).values.tolist()
    else:
        response = pd.DataFrame({'pred': pred, 'score': score}).values.tolist()
    logger.info(f'response: {response}')
    return response