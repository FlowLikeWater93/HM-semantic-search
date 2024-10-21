import pandas as pd
import pickle
import torch
import helpers
import sys
import encoder
from sklearn.model_selection import train_test_split


if sys.argv[1] == 'init' :
    print('### Generating tokens and Creating Vocabulary ###')
    # load dataset of H&M products
    df = pd.read_csv('../data/articles.csv')

    # select the three relevant columns and drop duplicate rows
    df = df.iloc[:,[2,19,24]].drop_duplicates()
    # drop row with null detail_desc values
    df = df.dropna(subset='detail_desc')
    # combine product name and product description columns
    df['clean_text'] = df.apply(lambda x: x.loc['prod_name'] + ' ' + x.loc['detail_desc'], axis=1)
    # standardize text
    df['clean_text'] = df['clean_text'].apply(lambda x: helpers.clean_text(x))
    # Generate tokens and create vocab
    vocab = helpers.generate_tokens(df['clean_text'].unique())
    ## test vocab
    print('testing vocab : ')
    print('index of word "bra" : {}'.format(vocab['bra']))
    print('index of word "microfibre" : {}'.format(vocab['microfibre']))
    print('index of word "shirt" : {}'.format(vocab['shirt']))
    print('index of word "pants" : {}'.format(vocab['pants']))
    print('index of word "cotton" : {}'.format(vocab['cotton']))
    print('### Done ###')

    print('### initializing and saving nlp model and model optimizer ###')
    # create model
    model = encoder.Encoder(len(vocab))
    # create adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # save model parameters
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, 'model0.pth')
    # save model metrics file
    with open('model_metrics.pickle', 'wb') as handle :
        pickle.dump({'training_loss': [], 'validation_loss': [], 'f1': [], 'accuracy': []}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # Save vocab file
    with open('vocab.pickle', 'wb') as handle :
        pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('### Done ###')


else :
    print('### Saving training and testing data ###')
    # load dataset of H&M products
    df = pd.read_csv('../data/articles.csv')
    # select the two relevant columns
    df = df.iloc[:,[2,19,24]]
    # drop row with null detail_desc values
    df = df.dropna(subset='detail_desc')
    # combine product name and product description columns
    df['clean_text'] = df.apply(lambda x: x.loc['prod_name'] + ' ' + x.loc['detail_desc'], axis=1)
    # standardize text
    df['clean_text'] = df['clean_text'].apply(lambda x: helpers.clean_text(x))
    # prepare labels
    df['clean_label'] = df['index_group_name'].apply(lambda x: ['Ladieswear', 'Baby/Children', 'Menswear', 'Sport', 'Divided'].index(x))

    # Train-Test split
    X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,3], df.iloc[:,4], test_size=0.15, random_state=42)

    # save datasets
    X_train.to_csv('../data/x_train.csv', index=False)
    X_test.to_csv('../data/x_test.csv', index=False)
    y_train.to_csv('../data/y_train.csv', index=False)
    y_test.to_csv('../data/y_test.csv', index=False)

    print('### Done ###')
