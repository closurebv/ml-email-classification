Data processing:

    Notebook files:

        1) 1_data_cleaning.ipynb: clean Dutch emails so it remains only the body
        2) 2_duplicates.ipynb: remove exact duplicates and also very similar emails considered as duplicates
        3) 3_outlier_detection.ipynb: remove emails considered as outliers
        4) 4_annotation_CS.ipynb: get labels from customer support
        5) 5_annotation.ipynb: manual annotation and generate final file with 3 languages

    The notebooks need to be execute in this order.
    
    Python files:

        1) connection_database.py: function to connect to Closure's database
        2) cleaning_functions.py: functions used in 1_data_cleaning.ipynb
        3) duplicates_removal.py: functions used in 2_duplicates.ipynb
        4) outlier_detection.py: functions used in 3_outlier_detection.ipynb
        5) annotation_functions.py: functions used in 4_annotation_CS.ipynb and 5_annotation.ipynb

    
Models fine-tuning:

    Notebook files:
        
        1) fine_tune_models.ipynb: fine-tune all models
        2) test_models.ipynb: test models already fine-tuned and give some metrics on test set
        
    Python files:
            
        1) model_tokenizer_loaders.py: functions to load the models and their tokenizers
        2) fine_tuning_functions.py: functions used in fine_tune_models.ipynb
        3) test_models_functions.py: functions used in test_models.ipynb
