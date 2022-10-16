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

class ReviewDataset(Dataset):
    '''
    `Dataset` wrapper for text 

    Parameters:
        input_text  (:class:`numpy.ndarray`, concatenated text with text_cols)
        labels (:class: list` or `numpy.ndarray`)

    Return:
        dict object with keys: [input_ids, attention_mask, labels]
    '''


    def __init__(self, input_text, labels, tokenizer, max_token_length):
        self.tokenizer = tokenizer
        self.labels = labels
        self.input_text = input_text
        self.max_token_length=max_token_length

    def __getitem__(self, idx):
        text  = str(self.labels[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_token_length,
            return_token_type_ids=False,
            # pad_to_max_length=True,
            padding='max_length',
            return_attention_mask=True,
            # truncation=True,
            return_tensors='pt', 
            )
        item = {}
        item['input_ids'] = encoding['input_ids'].flatten()
        item['attention_mask'] = encoding['attention_mask'].flatten()
        item['labels'] = torch.tensor(label, dtype=torch.long) 

        return item

    def __len__(self):
        return len(self.labels)

    def get_labels(self):
        """returns the label names for classification"""
        return self.labels

def create_data_loader(df, tokenizer, label_col, text_col, max_len, batch_size, drop_last):
    df = df.copy()
    try:
        labels = df[label_col].to_numpy()
    except:
        logger.info(f'No {label_col} col found in df.')
    
    dataset = ReviewDataset(
        input_text=df[text_col].to_numpy(),
        labels=labels,                    
        tokenizer=tokenizer,
        max_token_length=max_len)
        
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
        drop_last=drop_last # https://discuss.pytorch.org/t/error-expected-more-than-1-value-per-channel-when-training/26274/75
    )


def _get_train_val_dataloader(data_dir, tokenizer, label_col, text_col, max_len=512, batch_size=64, drop_last=False):
    '''
        data_dir: os.environ['SM_CHANNEL_TRAIN']
        tokenizer: tokenizer for Bert
        label_col: label column name
        batch_size: int
        drop_last: bool

        Return:
            train_data_loader, val_data_loader
    '''
    
    df = pd.read_csv(f'{data_dir}/train.csv')
        
    # test_size=0.2
    train_size = int(0.8 * len(df))
    val_size = len(df) - train_size

    # Create a list of indeces for all of the samples in the dataset.
    indeces = np.arange(0, len(df))
    # Shuffle the indeces randomly.
    random.shuffle(indeces)
    # Get a list of indeces for each of the splits.
    train_idx = indeces[0:train_size]
    val_idx = indeces[train_size:]

    train_data = df.iloc[train_idx].reset_index(drop=True)
    val_data = df.iloc[val_idx].reset_index(drop=True)

    # create dataloader for train/val
    train_data_loader = create_data_loader(train_data, tokenizer, label_col, text_col, max_len, batch_size, drop_last)
    val_data_loader = create_data_loader(val_data, tokenizer, label_col, text_col, max_len, batch_size, drop_last)

    return train_data_loader, val_data_loader


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
    
def train(args):
    
    train_loader, val_loader = _get_train_val_dataloader(args.data_dir, tokenizer, args.label_col, args.text_col, args.max_len, args.val_batch_size, args.drop_last)

    model = BertClassifier(args.n_classes)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss().to(device)

    for epoch in range(1, args.epochs + 1):
        model = model.to(device) 
        model.train()
        total_acc_train = 0
        total_loss_train = 0

        # train the model
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            # 0. zero grad
            model.zero_grad()
            # 1. forward pass
            outputs = model(input_ids, attention_mask)
            # 2. loss 
            loss = loss_fn(outputs, labels.long())

            total_loss_train += loss.item()
            acc = (outputs.argmax(dim=1) == labels).sum().item()
            total_acc_train += acc
            
            # 3. backward
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # 4. update step
            optimizer.step()

            
            if step % 100 == 0:
                logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, step * args.batch_size, len(train_loader.dataset),
                    100. * step / len(train_loader), loss.item()))

        # validate the model
        model.eval()
        total_acc_val = 0
        total_loss_val = 0
        best_accuracy = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                # 1. forward pass
                outputs = model(input_ids, attention_mask)
                # 2. loss 
                loss = loss_fn(outputs, labels.long())

                total_loss_val += loss.item()

                acc = (outputs.argmax(dim=1) == labels).sum().item()
                total_acc_val += acc

        val_acc = total_acc_val / len(val_loader.dataset)
        # save the model
        if val_acc > best_accuracy:
            save_model(model, args.model_dir)

        logger.info(
            f'Epochs: {epoch} | Train Loss: {total_loss_train / len(train_loader.dataset): .3f} | Train Accuracy: {total_acc_train / len(train_loader.dataset): .3f} | Val Loss: {total_loss_val / len(val_loader.dataset): .3f} | Val Accuracy: {val_acc: .3f}'
                    )

def save_model(model, model_dir):
    logger.info("Saving the model.")
    path = os.path.join(model_dir, "model.bin")
    torch.save(model.cpu().state_dict(), path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--val_batch_size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for validation (default: 64)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr", type=float, default=2e-5, metavar="LR", help="learning rate (default: 2e-5)"
    )
    
    parser.add_argument("--label_col", type=str, default='will_recommend')
    parser.add_argument("--text_col", type=str, default='review')
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--n_classes", type=int, default=2)
    parser.add_argument("--drop_last", type=bool, default=False)
    # Container environment
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])
              
    train(parser.parse_args())
  