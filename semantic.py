import torch
import pandas as pd
import numpy as np
import helpers
import pickle
import encoder
import sys


if sys.argv[1] == 'generate' :
    df = pd.read_csv('../data/articles.csv')
    df = df.iloc[:,[2,19,24]].drop_duplicates()
    df = df.dropna(subset='detail_desc')
    print(df.info())

    # Loading vocabulary
    with open('vocab.pickle', 'rb') as handle:
        vocab = pickle.load(handle)
    # Loading model params
    model_params = torch.load('model.pth', weights_only=True)
    # initializing model
    model = encoder.Encoder(len(vocab))
    model.load_state_dict(model_params['model'])

    # combine product name and product description columns
    df['clean_text'] = df.apply(lambda x: x.loc['prod_name'] + ' ' + x.loc['detail_desc'], axis=1)
    # standardize text
    df['clean_text'] = df['clean_text'].apply(lambda x: helpers.clean_text(x))
    # tokenize dataset
    dataset = helpers.tokenize(df['clean_text'].to_list(), vocab)
    # convert to torch tensor
    dataset_tensor = torch.tensor(dataset, dtype=torch.int32)

    # generating embeddings
    pointer = 0
    pointer_plus = 0
    embds = []
    batch_size = 4096

    print('\n#### Generating embeddings started ####')
    print('-----------------------------------------')
    with torch.no_grad():
        while pointer < dataset_tensor.shape[0] :
            print('Batch {} started ... '.format(int(pointer / batch_size) + 1))
            # prepare batch pointers
            if pointer + batch_size > dataset_tensor.shape[0] :
                pointer_plus = dataset_tensor.shape[0]
            else :
                pointer_plus = pointer + batch_size

            # pass batch to model
            embd = model.sentence_embedding(dataset_tensor[pointer:pointer_plus, :])
            # concatenate embeddings
            embds.append(embd)

            print('Batch {} Ended '.format(int(pointer / batch_size) + 1))
            print('-----------------------------------------')
            # increament batch pointer
            pointer += batch_size

        print('#### Generating embeddings ended ####')
        # concat results
        sentence_embeddings = torch.cat(embds, 0)
        # create a dataframe
        columns = ['embd'+str(i) for i in range(128)]
        final_df = pd.DataFrame(data=sentence_embeddings.detach().numpy(), columns=columns)
        # save final dataframe with embeddings and product name
        pd.concat([final_df.reset_index(drop=True), df.iloc[:,0].reset_index(drop=True)], axis=1).to_csv('../data/embeddings.csv', index=False)
        print('Saved embeddings successfully ...')



elif sys.argv[1] == 'cluster' :
    # load embeddings dataframe
    df_embeddings = pd.read_csv('../data/embeddings.csv')
    # convert embeddings to numpy
    embeddings = np.asarray(df_embeddings.iloc[:,:-1])

    # position of target item
    pos = int(sys.argv[2])
    print('#### Finding top-5 most similar items to our target item : {} - {} ####'.format(pos, df_embeddings.iloc[pos,-1]))

    # calculate manhattan distance between every item and our target item
    manhattan_dist = []
    for i in range(embeddings.shape[0]):
        if i == pos :
            manhattan_dist.append(100)
        else :
            manhattan_dist.append(np.absolute(embeddings[pos,:] - embeddings[i,:]).sum())


    # create a dataframe with product name and manhattan distance
    df_manhattan = pd.DataFrame(data={'prod_name': df_embeddings['prod_name'].to_list(), 'manhattan_dist': manhattan_dist})

    # sort values ascendingly by manhattan distance
    df_manhattan = df_manhattan.sort_values(by='manhattan_dist', ascending= True)

    # print top-5 items
    print(df_manhattan.iloc[:5, :])
    print('#### clustering ended ####')


elif sys.argv[1] == 'search' :
    # load embeddings dataframe
    df_embeddings = pd.read_csv('../data/embeddings.csv')
    # convert embeddings to numpy
    np_embeddings = np.asarray(df_embeddings.iloc[:,:-1])
    # Loading vocabulary
    with open('vocab.pickle', 'rb') as handle:
        vocab = pickle.load(handle)
    # Loading model params
    model_params = torch.load('model.pth', weights_only=True)
    # initializing model
    model = encoder.Encoder(len(vocab))
    model.load_state_dict(model_params['model'])

    # clean and tokenize query text
    query = sys.argv[2]
    print('#### Finding top-5 recommendations for our query : {} ####'.format(query))
    query = helpers.clean_text(query)
    query_tokens = helpers.tokenize([query], vocab)

    with torch.no_grad():
        query_embedding = model.sentence_embedding(torch.tensor(query_tokens, dtype=torch.int32)).detach().numpy()
        # un-comment if you need to classify the query text
        prediction = model(torch.tensor(query_tokens, dtype=torch.int32))
        print(torch.argmax(prediction, dim=1))

    # calculate manhattan distance between every item and our target item
    manhattan_dist = []
    for i in range(np_embeddings.shape[0]):
        manhattan_dist.append(np.absolute(query_embedding - np_embeddings[i,:]).sum())

    # create a dataframe with product name and manhattan distance
    df_manhattan = pd.DataFrame(data={'prod_name': df_embeddings['prod_name'].to_list(), 'manhattan_dist': manhattan_dist})

    # sort values ascendingly by manhattan distance
    df_manhattan = df_manhattan.sort_values(by='manhattan_dist', ascending= True)

    # print top-5 items
    print(df_manhattan.iloc[:5, :])
    print('#### search ended ####')



else :
    print('here....')
