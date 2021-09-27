#Code references:  
#[1] Joshi, P. (2020). Transfer Learning for NLP: Fine-Tuning BERT for Text Classification. https://www.analyticsvidhya.com/blog/2020/07/transfer-learning-for-nlp-fine-tuning-bert-for-text-classification/ 
#[2] Tran, C. (2021) Tutorial: Fine tuning BERT for Sentiment Analysis. https://skimai.com/fine-tuning-bert-for-sentiment-analysis/

from transformers import AutoModel, AutoTokenizer
from hyperopt import hp
import torch.nn as nn

def load_model(model_name):
    """ Import the pretrained model.
    num_labels is the number of labels you have for your classification (default is 2).
    
    Input:
        - model_name: str
            Name of the model to load. It has to be one of these strings:
            "BERT_CLS", "BERT_improved_CLS", "BERT_mean", "RoBERTa_CLS", 
            "RoBERTa_improved_CLS", "RoBERTa_mean", "RobBERT", "mBERT_test",
            "mBERT_all", "XLM-R_test", "XLM-R_all", "mBERT_all_tanh".
    """
    
    try:
        if "BERT_" in model_name:
            name = 'bert-base-cased'

        if "RoBERTa" in model_name:
            name = 'roberta-base'

        if "mBERT" in model_name:
            name = "bert-base-multilingual-cased"

        if "XLM-R" in model_name:    
            name = "xlm-roberta-base"

        if model_name == "RobBERT":
            name = "pdelobelle/robbert-v2-dutch-base"

        bert = AutoModel.from_pretrained(name, num_labels = 4, output_attentions=False,
                                                            output_hidden_states=False)
     
    # Raise an error if model_name was not correct
    except:
        print(""" Error: model_name has to be one of the following: "BERT_CLS", "BERT_improved_CLS", "BERT_mean", "RoBERTa_CLS", 
            "RoBERTa_improved_CLS", "RoBERTa_mean", "RobBERT", "mBERT_test",
            "mBERT_all", "XLM-R_test", "XLM-R_all", "mBERT_all_tanh" """)
              
    return(bert)


def load_tokenizer(model_name):
    """ Load model tokenizer.
    
    Input:
        - model_name: str
            Name of the model to load. It has to be one of these strings:
            "BERT_CLS", "BERT_improved_CLS", "BERT_mean", "RoBERTa_CLS", 
            "RoBERTa_improved_CLS", "RoBERTa_mean", "RobBERT", "mBERT_test",
            "mBERT_all", "XLM-R_test", "XLM-R_all", "mBERT_all_tanh".
    """
    
    try:
        if "BERT_" in model_name:
            name = 'bert-base-cased'

        if "RoBERTa" in model_name:
            name = 'roberta-base'

        if "mBERT" in model_name:
            name = "bert-base-multilingual-cased"

        if "XLM-R" in model_name:    
            name = "xlm-roberta-base"

        if model_name == "RobBERT":
            name = "pdelobelle/robbert-v2-dutch-base"
    
        tokenizer = AutoTokenizer.from_pretrained(name)
    
    # Raise an error if model_name was not correct
    except:
        print(""" Error: model_name has to be one of the following: "BERT_CLS", "BERT_improved_CLS", "BERT_mean", "RoBERTa_CLS", 
            "RoBERTa_improved_CLS", "RoBERTa_mean", "RobBERT", "mBERT_test",
            "mBERT_all", "XLM-R_test", "XLM-R_all", "mBERT_all_tanh" """)
              
    return(tokenizer)

class BERT_Arch(nn.Module):
    """ A class that defines BERT architecture.

    ...

    Attributes
    ----------
    bert : BERT pretrained model

    dropout: float between 0 and 1
    
    model_name: str
        Name of the model to load. It has to be one of these strings:
        "BERT_CLS", "BERT_improved_CLS", "BERT_mean", "RoBERTa_CLS", 
        "RoBERTa_improved_CLS", "RoBERTa_mean", "RobBERT", "mBERT_test",
        "mBERT_all", "XLM-R_test", "XLM-R_all", "mBERT_all_tanh".
    

    Methods
    -------
    forward(sent_id, mask, model_name)
        foward step that build the neural network architecture

    """

    def __init__(self, bert, dropout, model_name):
        """
        Parameters
        ----------
        bert : BERT pretrained model

        dropout: float between 0 and 1
        
        model_name: str
            Name of the model to load. It has to be one of these strings:
            "BERT_CLS", "BERT_improved_CLS", "BERT_mean", "RoBERTa_CLS", 
            "RoBERTa_improved_CLS", "RoBERTa_mean", "RobBERT", "mBERT_test",
            "mBERT_all", "XLM-R_test", "XLM-R_all", "mBERT_all_tanh".
        """

        super(BERT_Arch, self).__init__()

        self.bert = bert 
        self.dropout = dropout

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # ReLU activation function
        self.relu =  nn.ReLU()
        
        # Tanh activation function
        self.tanh =  nn.Tanh()
        
        # Linear layer CLS-pooling
        self.fc0 = nn.Linear(768,4)
        
        # Linear layer 1
        self.fc1 = nn.Linear(768,512)

        # Linear layer 2 (Output layer)
        self.fc2 = nn.Linear(512,4) 

        # Softmax activation function
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask, model_name):
        """ Apply corresponding model with the chosen pooling technique.

        Parameters
        ----------
        sent_id: BERT sentence IDs 
        
        mask: BERT mask
        
        model_name: str
            Name of the model to load. It has to be one of these strings:
            "BERT_CLS", "BERT_improved_CLS", "BERT_mean", "RoBERTa_CLS", 
            "RoBERTa_improved_CLS", "RoBERTa_mean", "RobBERT", "mBERT_test",
            "mBERT_all", "XLM-R_test", "XLM-R_all", "mBERT_all_tanh".
        """

        # Get the row output of bert as a dictionary.
        model_result = self.bert(sent_id, attention_mask=mask, return_dict=True)
            
        #If this is not mean-pooling
        if "CLS" in model_name:
            # Keep only the pooled CLS token (after linear layer and tanh activation)
            token_embeddings = model_result.pooler_output
            
            #If this is CLS-pooling (not improved)
            if model_name == "BERT_CLS" or model_name == "RoBERTa_CLS":
                #Linear layer CLS-pooling
                x = self.fc0(token_embeddings)
                
                # Dropout layer
                x = self.dropout(x)
                
                # Softmax activation function
                x = self.softmax(x)
        
        #If this is mean-pooling
        if "CLS" not in model_name :
            # Keep only the last hidden state.
            token_embeddings = model_result.last_hidden_state
            # Make the average of all the tokens of the last hidden state.
            token_embeddings = token_embeddings.sum(axis=1) / mask.sum(axis=-1).unsqueeze(-1)
        
        #If this is not a CLS-pooling and not a Tanh activation
        if model_name != "BERT_CLS" and model_name != "RoBERTa_CLS" and model_name != "mBERT_all_tanh":
            # First linear layer
            x = self.fc1(token_embeddings)

            # ReLU activation function
            x = self.relu(x)

            # Dropout layer
            x = self.dropout(x)

            # Last linear layer
            x = self.fc2(x)

            # Softmax activation function
            x = self.softmax(x)
            
         if model_name == "mBERT_all_tanh":
            # First linear layer
            x = self.fc1(token_embeddings)

            # ReLU activation function
            x = self.relu(x)
            
            # Tanh activation function
            x = self.tanh(x)

            # Dropout layer
            x = self.dropout(x)

            # Last linear layer
            x = self.fc2(x)

            # Softmax activation function
            x = self.softmax(x)

        return(x)
    
def models_hyperparameters(model_name):
    """ Define TPE domain space and maximum number of trials of the model.
    
    Input:
        - model_name: str
            Name of the model to load. It has to be one of these strings:
            "BERT_CLS", "BERT_improved_CLS", "BERT_mean", "RoBERTa_CLS", 
            "RoBERTa_improved_CLS", "RoBERTa_mean", "RobBERT", "mBERT_test",
            "mBERT_all", "XLM-R_test", "XLM-R_all", "mBERT_all_tanh".
    """
    
    name = 'TPE_' + model_name
    
    #If this is CLS-pooling
    if "CLS" in model_name:
        space = {
        "lr": hp.choice("lr", [5*10**(-5), 10**(-4), 5*10**(-4), 10**(-3), 5*10**(-3)]),
        "dropout": hp.choice("dropout", [0.2, 0.3, 0.4, 0.5]),
        "epochs": hp.choice("epochs", [350]),
        "folder_name": hp.choice("folder_name", [name])
        }

        nb_trials = 5
        
    #If this is BERT_mean or RoBERTa_mean
    if "mean" in model_name:   
        space = {
            "lr": hp.choice("lr", [5*10**(-5), 10**(-4), 5*10**(-4), 10**(-3), 5*10**(-3)]),
            "dropout": hp.choice("dropout", [0.2, 0.3, 0.4, 0.5]),
            "epochs": hp.choice("epochs", [100]),
            "folder_name": hp.choice("folder_name", [name])
        }

        nb_trials = 15 
           
    #If model_name is RobBERT or mBERT_test or XLM-R_test
    if model_name == "RobBERT" or "test" in model_name:
        space = {
            "lr": hp.choice("lr", [5*10**(-5), 10**(-4), 5*10**(-4), 10**(-3), 5*10**(-3)]),
            "dropout": hp.choice("dropout", [0.2, 0.3, 0.4, 0.5]),
            "epochs": hp.choice("epochs", [150]),
            "folder_name": hp.choice("folder_name", [name])
        }

        nb_trials = 10
        
    if "mBERT_all" in model_name:
        space = {
            "lr": hp.choice("lr", [5*10**(-5), 10**(-4), 5*10**(-4), 10**(-3), 5*10**(-3)]),
            "dropout": hp.choice("dropout", [0.2, 0.3, 0.4, 0.5]),
            "epochs": hp.choice("epochs", [60]),
            "folder_name": hp.choice("folder_name", [name])
        }

        nb_trials = 10
        
    if model_name == "XLM-R_all":
        space = {
            "lr": hp.choice("lr", [5*10**(-5), 10**(-4), 5*10**(-4), 10**(-3), 5*10**(-3)]),
            "dropout": hp.choice("dropout", [0.2, 0.3, 0.4, 0.5]),
            "epochs": hp.choice("epochs", [50]),
            "folder_name": hp.choice("folder_name", [name])
        }

        nb_trials = 10
    
    # Raise an error if model_name was not correct
    try:
        space
    except:
        print(""" Error: model_name has to be one of the following: "BERT_CLS", "BERT_improved_CLS", "BERT_mean", "RoBERTa_CLS", 
            "RoBERTa_improved_CLS", "RoBERTa_mean", "RobBERT", "mBERT_test",
            "mBERT_all", "XLM-R_test", "XLM-R_all", "mBERT_all_tanh" """)
              
    return(space, nb_trials, model_name)
