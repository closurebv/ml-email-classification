# ml-email-classification

This project classifies the received emails into four categories: "case solved", "person is not found", "more information are required", "irrelevant for case (automatic)". 

The code can be found in src and the documentation in doc. The documentation describes the files in src and the order in which to use them.
The fine-tuned models are: BERT, RoBERTa, mBERT (multilingual BERT), XLM-R (multilingual RoBERTa) and RobBERT (Dutch RoBERTa) with different pooling techniques. 

The optimal model is mBERT with mean-pooling in a translate-train-all setting (fine-tune and test in Dutch, translated-English and translated-French) 
and a total accuracy on the three languages of 95% on test set.


Associated paper: 'IMECABERT: An Imbalanced Multi-class Email Classification Approach for customer support using BERT models' (Bout, 2021)


