#Code references:  
#[1] Joshi, P. (2020). Transfer Learning for NLP: Fine-Tuning BERT for Text Classification. https://www.analyticsvidhya.com/blog/2020/07/transfer-learning-for-nlp-fine-tuning-bert-for-text-classification/  
#[2] Tran, C. (2021) Tutorial: Fine tuning BERT for Sentiment Analysis. https://skimai.com/fine-tuning-bert-for-sentiment-analysis/

from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import pickle
import torch
import model_tokenizer_loaders
import torch.nn as nn
import pandas as pd
# specify GPU
device = torch.device("cuda")

def split_dataset(data, model_name):
    """ Split the dataset into three datasets such as 
    Training set = 70%
    Validation set = 20%
    Test set = 10% 
    The datasets are stratified (they have the same proportions per each label 
    as in the original dataset).
    The random state allows reproducibility.
    
    Inputs:
        - data: DataFrame that must contains:
            - a content_en column containing the body of the email in English
            - a only_body column containing the body of the email in Dutch
            - a content_fr column containing the body of the email in French
            - a manual_label column containing the associated label (0, 1, 2, 3)
        - model_name: str
            Name of the model to load. It has to be one of these strings:
            "BERT_CLS", "BERT_improved_CLS", "BERT_mean", "RoBERTa_CLS", 
            "RoBERTa_improved_CLS", "RoBERTa_mean", "RobBERT", "mBERT_test",
            "mBERT_all", "XLM-R_test", "XLM-R_all", "mBERT_all_tanh".
    """

    train_text, temp_text, train_labels, temp_labels = train_test_split(data[['content_en', 'only_body', 'content_fr']], data['manual_label'], 
                                                                    random_state=0, 
                                                                    test_size=0.3, 
                                                                    stratify=data['manual_label'])


    val_text, test_text, val_labels, test_labels = train_test_split(temp_text, temp_labels, 
                                                                  random_state=0, 
                                                                  test_size=1/3,
                                                                  stratify=temp_labels)
    
    #If this is an English-based model
    if "test" not in model_name and "all" not in model_name and model_name != "RobBERT":
        train_text = train_text["content_en"]
        val_text = val_text["content_en"]
        test_text = test_text["content_en"]
    
    #If this is a multilingual fine-tuned only in Dutch
    if "test" in model_name:
        train_text = train_text["only_body"]
        val_text = val_text["only_body"]
        
    if model_name == "RobBERT":
        train_text = train_text["only_body"]
        val_text = val_text["only_body"]
        test_text = test_text["only_body"]
    
    #If this is a multilingual fine-tuned with the 3 languages
    if "all" in model_name:
        #Regroup the columns (3 languages) as one
        train_text = train_text.unstack().reset_index(drop=True)
        train_labels = pd.concat([train_labels]*3, ignore_index=True)

        val_text = val_text.unstack().reset_index(drop=True)
        val_labels = pd.concat([val_labels]*3, ignore_index=True)

        #Shuffle datasets so the languages are shuffled
        train_text = train_text.sample(frac=1, random_state=0)
        train_labels = train_labels.sample(frac=1, random_state=0)

        val_text = val_text.sample(frac=1, random_state=0)
        val_labels = val_labels.sample(frac=1, random_state=0)

    return(train_text, train_labels, val_text, val_labels, test_text, test_labels)

def tokenize_train_val(tokenizer, train_text, val_text):
    """ Tokenize training and validation sets.

    We set the max_length to 250 instead of the default 512, so the model can be 
    faster and as the majority of emails are composed of less than 250 words. If
    they are longer, they will be truncated.

    Input: 
        - tokenizer : Output of load_tokenizer()
        - train_text: Pandas Series
            1st output of split_dataset()
        - val_text: Pandas Series
            3rd output of split_dataset()
    """

    # Tokenize and encode sequences in the training set
    tokens_train = tokenizer.batch_encode_plus(
      train_text.tolist(),
      max_length = 250,
      truncation=True,
      padding=True,
      add_special_tokens=True, 
      return_attention_mask=True, 
      return_tensors='pt'
    )

    # Tokenize and encode sequences in the validation set
    tokens_val = tokenizer.batch_encode_plus(
      val_text.tolist(),
      max_length = 250,
      truncation=True,
      padding=True,
      add_special_tokens=True, 
      return_attention_mask=True, 
      return_tensors='pt'
    )
        
    return(tokens_train, tokens_val)
    
def tokenize_test(tokenizer, test_text):
    """ Tokenize test set.

    We set the max_length to 250 instead of the default 512, so the model can be 
    faster and as the majority of emails are composed of less than 250 words. If
    they are longer, they will be truncated.

    Input: 
        - tokenizer : Output of load_tokenizer()
        - test_text: Pandas Series
            5th output of split_dataset()
    """

    #If model is unilingual
    if type(test_text) == pd.core.series.Series:
        # Tokenize and encode sequences in the test set
        tokens_test = tokenizer.batch_encode_plus(
          test_text.tolist(),
          max_length = 250,
          truncation=True,
          padding=True,
          add_special_tokens=True, 
          return_attention_mask=True, 
          return_tensors='pt'
        )
        
        return(tokens_test)
    
    #If model is multilingual type(test_text) == DataFrame
    else:
        # tokenize and encode sequences in the test set
        tokens_test_en = tokenizer.batch_encode_plus(
            test_text["content_en"].tolist(),
            max_length = 250,
            truncation=True,
            padding=True,
            add_special_tokens=True, 
            return_attention_mask=True, 
            return_tensors='pt'
        )
    
        tokens_test_nl = tokenizer.batch_encode_plus(
            test_text["only_body"].tolist(),
            max_length = 250,
            truncation=True,
            padding=True,
            add_special_tokens=True, 
            return_attention_mask=True, 
            return_tensors='pt'
        )

        tokens_test_fr = tokenizer.batch_encode_plus(
            test_text["content_fr"].tolist(),
            max_length = 250,
            truncation=True,
            padding=True,
            add_special_tokens=True, 
            return_attention_mask=True, 
            return_tensors='pt'
        )
        
        return(tokens_test_nl, tokens_test_en, tokens_test_fr)

def to_tensor(tokenizer, train_text, val_text, train_labels, val_labels):
    """ Convert lists to tensors for training and validation sets. 

    Input: 
        - tokenizer : Output of load_tokenizer()
        - train_text: Pandas Series
            1st output of split_dataset()
        - val_text: Pandas Series
            3rd output of split_dataset()
        - train_labels: Pandas Series
            2nd output of split_dataset()
        - val_labels: Pandas Series
            4th output of split_dataset()
    """

    tokens_train, tokens_val = tokenize_train_val(tokenizer, train_text, val_text)

    train_seq = torch.tensor(tokens_train['input_ids'])
    train_mask = torch.tensor(tokens_train['attention_mask'])
    train_y = torch.tensor(train_labels.tolist())

    val_seq = torch.tensor(tokens_val['input_ids'])
    val_mask = torch.tensor(tokens_val['attention_mask'])
    val_y = torch.tensor(val_labels.tolist())

    return(train_seq, train_mask, train_y, val_seq, val_mask, val_y)

def to_tensor_test(tokenizer, test_text, test_labels):
    """ Convert lists to tensors for test set. 

    Input: 
        - tokenizer : Output of load_tokenizer()
        - test_text: Pandas Series
            5th output of split_dataset()
        - test_labels: Pandas Series
            6th output of split_dataset()
    """
    
    #If model is unilingual
    if type(test_text) == pd.core.series.Series:
        tokens_test = tokenize_test(tokenizer, test_text)
        
        test_seq = torch.tensor(tokens_test['input_ids'])
        test_mask = torch.tensor(tokens_test['attention_mask'])
        test_y = torch.tensor(test_labels.tolist())
        
        return(test_seq, test_mask, test_y)
    
    else:
        tokens_test_nl, tokens_test_en, tokens_test_fr = tokenize_test(tokenizer, test_text)
        
        test_seq_nl = torch.tensor(tokens_test_nl['input_ids'])
        test_mask_nl = torch.tensor(tokens_test_nl['attention_mask'])

        test_seq_en = torch.tensor(tokens_test_en['input_ids'])
        test_mask_en = torch.tensor(tokens_test_en['attention_mask'])

        test_seq_fr = torch.tensor(tokens_test_fr['input_ids'])
        test_mask_fr = torch.tensor(tokens_test_fr['attention_mask'])

        test_y = torch.tensor(test_labels.tolist())

        return(test_seq_nl, test_mask_nl, test_seq_en, test_mask_en, test_seq_fr, test_mask_fr, test_y)

def loader(tokenizer, train_text, val_text, test_text, train_labels, val_labels):
    """ Get training and validation tensors, samplers and dataloaders.
    We define the batch size here to 32.

    Input: 
        - tokenizer : Output of load_tokenizer()
        - train_text: Pandas Series
            1st output of split_dataset()
        - val_text: Pandas Series
            3rd output of split_dataset()
        - test_text: Pandas Series
            5th output of split_dataset()
        - train_labels: Pandas Series
            2nd output of split_dataset()
        - val_labels: Pandas Series
            4th output of split_dataset()
    """

    train_seq, train_mask, train_y, val_seq, val_mask, val_y = to_tensor(tokenizer, train_text, val_text, train_labels, val_labels)

    # Define a batch size
    batch_size = 32

    #----------------------------------Training----------------------------------
    # Wrap tensors
    train_data = TensorDataset(train_seq, train_mask, train_y)

    # Sampler for sampling the data during training
    train_sampler = RandomSampler(train_data)

    # DataLoader for train set
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    #----------------------------------Validation---------------------------------
    val_data = TensorDataset(val_seq, val_mask, val_y)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler = val_sampler, batch_size=batch_size)

    return(train_data, train_sampler, train_dataloader, val_data, val_sampler, val_dataloader)

def get_optimizer(lr, model):
    """ AdamW optimizer.

    Inputs:
        - lr: float
            learning rate will be fine-tuned with the TPE algorithm
        - model: Output of BERT_Arch class
            BERT model and its architecture
    """

    optimizer = AdamW(model.parameters(),
                    lr = lr,
                    eps = 1e-8)  

    return(optimizer)        

def get_scheduler(epochs, optimizer, train_dataloader):
    """ Linear Warmup scheduler.

    Inputs:
        - epochs: integer
            number of epochs used to train the model
        - optimizer: Output of get_optimizer()
        - train_dataloader: DataLoader
            3rd output of loader()
    """

    # The number of epochs is a parameter that will be defined in the TPE domain.
    epochs = epochs

    # Total number of training steps is [number of batches] x [number of epochs]. 
    # (Note that this is not the same as the number of training samples).
    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                              num_warmup_steps = 0, # Default value in run_glue.py
                                              num_training_steps = total_steps)

    return(scheduler)

def get_weights(train_labels):
    """ Compute the weights for each labels (we are in an imbalanced class setting).
    Pass the weights as a parameter in the loss function, the negative log-likelihood
    and returns it. The weights are computed on the training set because all sets 
    are statified.
    
    Input: 
        - train_labels: Pandas Series
            2nd output of split_dataset()
    """

    # Compute the class weights
    class_weights = compute_class_weight('balanced', np.unique(train_labels), train_labels)

    # Converting list of class weights to a tensor
    weights= torch.tensor(class_weights,dtype=torch.float)

    # Push to GPU
    weights = weights.to(device)

    # Define the loss function
    cross_entropy  = nn.NLLLoss(weight=weights) 

    return(cross_entropy)

def evaluate(y_pred, y_true):
    """ This function is used during training and validation to be able to 
    follow the evolution of these metrics amount the epochs. 

    It calculates true positive (TP), true negative (TN), false positive (FP),
    false negative (FN), true positive rate (TPR), true negative rate (TNR),
    false positive rate (FPR), false negative rate (FNR) and then accuracy.

    Inputs:
      - y_pred: list
          predicted probabilities for each label
      - y_true: list
          real labels
    """

    y_true = y_true.float().cpu().numpy() 

    # Get predicted labels (labels with the highest probability)
    new_y_pred = []
    for i in range(0, len(y_pred)):
        new_y_pred.append(np.argmax(y_pred[i]))
    
    TP = np.sum((y_true + new_y_pred) == 2)
    TN = np.sum((y_true + new_y_pred) == 0)
    FN = np.sum((new_y_pred - y_true) < 0)
    FP = np.sum((y_true - new_y_pred) < 0)
    
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    TNR = TN / (TN + FP)
    FNR = FN / (FN + TP)
    
    Accuracy = (TP + TN) / (TP + TN + FP + FN)

    return TPR, TNR, FNR, FPR, Accuracy

def train(model, train_dataloader, cross_entropy, optimizer, scheduler, model_name):
    """ Train the model.

    Inputs:
        - model: Output of BERT_Arch class
            BERT model and its architecture
        - train_dataloader: DataLoader
            3rd output of loader()
        - cross_entropy: Output of get_weights()
        - optimizer: Output of get_optimizer()
        - scheduler: Output of get_scheduler()
        - model_name: str
                Name of the model to load. It has to be one of these strings:
                "BERT_CLS", "BERT_improved_CLS", "BERT_mean", "RoBERTa_CLS", 
                "RoBERTa_improved_CLS", "RoBERTa_mean", "RobBERT", "mBERT_test",
                "mBERT_all", "XLM-R_test", "XLM-R_all", "mBERT_all_tanh".
    """

    model.train()

    total_loss, total_accuracy = 0, 0

    L = len(train_dataloader.dataset)

    # Empty list to save model predictions and metrics
    total_preds=[]
    losses = []

    TPR = []
    TNR = []
    FNR = []
    FPR = []
    Accuracy = []

    # Iterate over batches
    for step, batch in enumerate(train_dataloader):

        # Print progress update after every 50 batches.
        if step % 50 == 0 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))

        # Clear previously calculated gradients 
        model.zero_grad()

        # Push the batch to GPU
        batch = tuple(r.to(device) for r in batch)

        sent_id, mask, labels = batch

        # Get model predictions for the current batch
        preds = model(sent_id, mask, model_name)

        # Compute the loss between actual and predicted values
        loss = cross_entropy(preds, labels)

        # Add on to the total loss
        total_loss = total_loss + loss.item()

        # Backward pass to calculate the gradients
        loss.backward()
        losses.append(loss.item())

        # Clip the the gradients to 1.0. It helps in preventing the exploding gradient problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update parameters
        optimizer.step()

        # Update the learning rate
        scheduler.step()

        # Model predictions are stored on GPU. So, push it to CPU.
        preds=preds.detach().cpu().numpy()

        # Append the model predictions
        total_preds.append(preds)

        # Evaluate the model to get metric results
        TPR_, TNR_, FNR_, FPR_, Accuracy_ = evaluate(preds, labels)
        TPR.append(TPR_)
        TNR.append(TNR_)
        FNR.append(FNR_)
        FPR.append(FPR_)
        Accuracy.append(Accuracy_)

    # Compute the training loss of the epoch
    avg_loss = total_loss / len(train_dataloader)

    # Predictions are in the form of (no. of batches, size of batch, no. of classes).
    # Reshape the predictions in form of (number of samples, no. of classes).
    total_preds  = np.concatenate(total_preds, axis=0)

    # Return the loss, predictions and metrics
    return avg_loss, total_preds, np.mean(TPR), np.mean(TNR), np.mean(FNR), np.mean(FPR), np.mean(Accuracy)

def validation(model, train_dataloader, val_dataloader, cross_entropy, model_name):
    """ Validate the model.

    Inputs:
        - model: Output of BERT_Arch class
            BERT model and its architecture
        - train_dataloader: DataLoader
            3rd output of loader()
        - val_dataloader: DataLoader
            6th output of loader()
        - cross_entropy: Output of get_weights()
        - model_name: str
                Name of the model to load. It has to be one of these strings:
                "BERT_CLS", "BERT_improved_CLS", "BERT_mean", "RoBERTa_CLS", 
                "RoBERTa_improved_CLS", "RoBERTa_mean", "RobBERT", "mBERT_test",
                "mBERT_all", "XLM-R_test", "XLM-R_all", "mBERT_all_tanh".
    """

    print("\nEvaluating...")

    # Deactivate dropout layers
    model.eval()

    total_loss, total_accuracy = 0, 0

    L = len(train_dataloader.dataset)

    # Empty list to save the model predictions and metrics
    total_preds = []
    losses = []

    TPR = []
    TNR = []
    FNR = []
    FPR = []
    Accuracy = []

    # Iterate over batches
    for step, batch in enumerate(val_dataloader):

        # Print progress update every 50 batches.
        if step % 50 == 0 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(val_dataloader)))

        # Push the batch to GPU
        batch = [t.to(device) for t in batch]

        sent_id, mask, labels = batch

        # Deactivate autograd
        with torch.no_grad():

            # Model predictions
            preds = model(sent_id, mask, model_name)

            # Compute the validation loss between actual and predicted values
            loss = cross_entropy(preds,labels)

            total_loss = total_loss + loss.item()

            preds = preds.detach().cpu().numpy()

            total_preds.append(preds)

        losses.append(loss.item())

    TPR_, TNR_, FNR_, FPR_, Accuracy_ = evaluate(preds, labels)
    TPR.append(TPR_)
    TNR.append(TNR_)
    FNR.append(FNR_)
    FPR.append(FPR_)
    Accuracy.append(Accuracy_)

    # Compute the validation loss of the epoch
    avg_loss = total_loss / len(val_dataloader) 

    # Reshape the predictions in form of (number of samples, no. of classes)
    total_preds  = np.concatenate(total_preds, axis=0)

    return avg_loss, total_preds, np.mean(TPR), np.mean(TNR), np.mean(FNR), np.mean(FPR), np.mean(Accuracy)

def fine_tuning_model(epochs, lr, dropout, folder_name, train_text, val_text, test_text, train_labels, val_labels, model_name):
    """ Fine-tune the model with hyperparameters: epochs, lr and dropout.
    Save in three different files: training and validation losses and the number 
    of epochs.

    Inputs:
        - epochs: integer
            number of epochs used to train the model
        - lr: float
            learning rate will be fine-tuned with the TPE algorithm
        - dropout: float between 0 and 1
        - folder_name: str
            name of the repository to save the model files
        - train_text: Pandas Series
            1st output of split_dataset()
        - val_text: Pandas Series
            3rd output of split_dataset()
        - test_text: Pandas Series
            5th output of split_dataset()
        - train_labels: Pandas Series
            2nd output of split_dataset()
        - val_labels: Pandas Series
            4th output of split_dataset()
        - model_name: str
            Name of the model to load. It has to be one of these strings:
            "BERT_CLS", "BERT_improved_CLS", "BERT_mean", "RoBERTa_CLS", 
            "RoBERTa_improved_CLS", "RoBERTa_mean", "RobBERT", "mBERT_test",
            "mBERT_all", "XLM-R_test", "XLM-R_all", "mBERT_all_tanh".
    """

    # Set initial loss to infinite
    best_valid_loss = float('inf')

    # Empty lists to store training and validation loss of each epoch and metrics
    train_losses=[]
    valid_losses=[]
    all_epochs = []
    all_accuracy = []
    all_accuracy_train = []
    all_precision = []
    all_precision_train = []
    all_recall = []
    all_recall_train = []
    all_f1 = []
    all_f1_train = []

    # Load model and tokenizer
    bert = model_tokenizer_loaders.load_model(model_name)
    tokenizer = model_tokenizer_loaders.load_tokenizer(model_name)
    # Pass the pre-trained BERT to our define architecture
    model = model_tokenizer_loaders.BERT_Arch(bert, dropout, model_name)

    # Push the model to GPU
    model = model.to(device)

    train_data, train_sampler, train_dataloader, val_data, val_sampler, val_dataloader = loader(tokenizer, train_text, val_text, test_text, train_labels, val_labels)

    # Freeze BERT parameters
    for param in bert.parameters():
        param.requires_grad = False

    optimizer = get_optimizer(lr, model)
    scheduler = get_scheduler(epochs, optimizer, train_dataloader)
    cross_entropy = get_weights(train_labels)

    # For each epoch
    for epoch in range(epochs):

        print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))

        all_epochs.append(epoch + 1)

        # Training model
        train_loss, _, TPR_train, TNR_train, FNR_train, FPR_train, Accuracy_train = train(model, train_dataloader, cross_entropy, optimizer, scheduler, model_name)

        # validation model
        valid_loss, _, TPR, TNR, FNR, FPR, Accuracy = validation(model, train_dataloader, val_dataloader, cross_entropy, model_name)


        # Save the best model (model with the lower validation loss)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            name = folder_name + '/saved_weights_lr{:}_dropout{:}_epochs{:}.pt'.format(lr, dropout, epochs)
            torch.save(model.state_dict(), name)

        # Append training and validation loss
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        # Append training and validation accuracies
        all_accuracy.append(Accuracy)
        all_accuracy_train.append(Accuracy_train)

        print(f'\nTraining Loss: {train_loss:.3f}')
        print(f'Validation Loss: {valid_loss:.3f}')
        print('Train :')
        print("Metrics : TPR {:.3f} TNR {:.3f} FNR {:.3f} FPR {:.3f} Accuracy {:.3f}".format(TPR_train, TNR_train, FNR_train, FPR_train, Accuracy_train))
        print(' ')
        print('Val :')
        print("Metrics : TPR {:.3f} TNR {:.3f} FNR {:.3f} FPR {:.3f} Accuracy {:.3f}".format(TPR, TNR, FNR, FPR, Accuracy))

    # Save in pickle files, training and validation losses and the number of epochs
    
    name = folder_name + '/train_losses_lr{:}_dropout{:}_epochs{:}.txt'.format(lr, dropout, epochs)
    with open(name, "wb") as f:   
        pickle.dump(train_losses, f)
    
    name = folder_name + '/val_losses_lr{:}_dropout{:}_epochs{:}.txt'.format(lr, dropout, epochs)
    with open(name, "wb") as f: 
        pickle.dump(valid_losses, f)
    
    name = folder_name + '/all_epochs_lr{:}_dropout{:}_epochs{:}.txt'.format(lr, dropout, epochs)
    with open(name, "wb") as f: 
        pickle.dump(all_epochs, f)

    return(model)
