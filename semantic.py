import torch
import pandas as pd
import numpy as np
import helpers
import pickle
import encoder
import sys
from sklearn.cluster import KMeans


if len(sys.argv) == 1 :
    print('########## Guide ##########')
    print('1- python semantic.py generate-index')
    print('1.1- Generates sentence embeddings for all of the products in the database')
    print('1.2- Builds a Kmeans clustring model')
    print('1.3- Saves embeddings, cluster labels and kmeans model')
    print('2- python semantic.py cluster <product_position>')
    print('2.1- Find top-n most similar items to the item at position <product_position>')
    print('3- python semantic.py search <\'search sentence\'>')
    print('3.1- Converts search sentence to sentence embedding')
    print('3.2- Uses kmeans model to predict cluster label')
    print('3.3- Retrieves all items with the same cluster label')
    print('3.4- Calculates manhattan distance and return top-n most similar items')



elif sys.argv[1] == 'generate+index' :
    # load original items dataframe
    df_org = pd.read_csv('../data/articles.csv')
    # select relevant columns and drop duplicate rows
    df = df_org.iloc[:,[2,19,24]].drop_duplicates()
    # drop rows with null item description values
    df = df.dropna(subset='detail_desc')
    # add item id column from original dataframe
    df = pd.concat([df,df_org.loc[df.index,'article_id']], axis=1)
    print(df.info())
    # delete original items dataframe
    del df_org

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
    batch_size = 1024

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
        # free up disk space
        del embds
        del dataset_tensor
        del dataset
        del vocab
        del model_params
        # clustring users
        kmeans = KMeans(n_clusters=50, random_state=42, n_init="auto").fit(sentence_embeddings.detach().numpy())
        # concatenate embeddings with cluster labels
        np_sentence_embeddings = np.concatenate((sentence_embeddings.detach().numpy(), np.array([kmeans.labels_]).T), axis=1)
        # create new dataframe
        columns = ['embd'+str(i) for i in range(128)]
        columns.append('kmeans_label')
        final_df = pd.DataFrame(data=np_sentence_embeddings, columns=columns)
        # save final dataframe with embeddings and product name
        pd.concat([final_df.reset_index(drop=True), df.iloc[:,3].reset_index(drop=True)], axis=1).to_csv('embeddings.csv', index=False)
        # save kmeans model
        with open('semantic_kmeans.pickle', 'wb') as handle :
            pickle.dump(kmeans, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('Saved embeddings successfully ...')



elif sys.argv[1] == 'cluster' :
    print('#### clustering started ####\n')
    # position of target item
    pos = int(sys.argv[2])
    # load items dataframe
    df_articles = pd.read_csv('../data/articles.csv')
    # load embeddings dataframe
    df_embeddings = pd.read_csv('embeddings.csv')
    # extract target vector
    target = df_embeddings.iloc[pos,:-2].to_numpy()
    target_id = df_embeddings.iloc[pos,-1]
    # select embeddings in the same cluster only
    df_embeddings = df_embeddings.loc[(df_embeddings['kmeans_label']==df_embeddings.iloc[pos,-2]) & (df_embeddings['article_id'] != target_id)]
    # convert embeddings to numpy
    embeddings_cluster = np.asarray(df_embeddings.iloc[:,:-2])
    print('#### Finding top-5 most similar items to our target item : {} - {} ####'.format(pos, target_id))
    print('Target product name : {}\nTarget product description : {}\n'.format(df_articles[df_articles['article_id']==target_id].iloc[0,2], df_articles[df_articles['article_id']==target_id].iloc[0,24]))
    # calculate manhattan distance between every item and our target item
    manhattan_dist = []
    for i in range(embeddings_cluster.shape[0]):
        manhattan_dist.append(np.absolute(target - embeddings_cluster[i,:]).sum())
    # create a dataframe with product name and manhattan distance
    df_manhattan = pd.DataFrame(data={'article_id': df_embeddings['article_id'].to_list(), 'manhattan_dist': manhattan_dist})
    # sort values ascendingly by manhattan distance
    df_manhattan = df_manhattan.sort_values(by='manhattan_dist', ascending= True)
    # print top-5 items
    for i in range(5):
        print('Top ({})\nproduct id : {}\nproduct name : {}\nproduct description : {}\n'.format(i+1, df_manhattan.iloc[i, 0], df_articles[df_articles['article_id']==df_manhattan.iloc[i, 0]].iloc[0,2], df_articles[df_articles['article_id']==df_manhattan.iloc[i, 0]].iloc[0,24]))
    print('#### clustering ended ####')


elif sys.argv[1] == 'search' :
    # define list of item classes
    item_classes = ['Ladieswear', 'Baby/Children', 'Menswear', 'Sport', 'Divided']
    # load embeddings dataframe
    df_embeddings = pd.read_csv('embeddings.csv')
    # load items dataframe
    df_articles = pd.read_csv('../data/articles.csv')
    # Loading kmeans model
    with open('semantic_kmeans.pickle', 'rb') as handle:
        kmeans = pickle.load(handle)
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
    query = helpers.clean_text(query)
    query_tokens = helpers.tokenize([query], vocab)
    # generate sentence embedding
    with torch.no_grad():
        query_embedding = model.sentence_embedding(torch.tensor(query_tokens, dtype=torch.int32)).detach().numpy()
        # un-comment if you need to classify the query text
        prediction = model(torch.tensor(query_tokens, dtype=torch.int32))
        print('\nPredicted class : {}'.format(item_classes[torch.argmax(prediction, dim=1)[0]]))

    print('\n#### Finding top-5 recommendations for our query ####\n{}\n'.format(query))
    # predict label pf query text
    query_label = kmeans.predict(query_embedding)
    # select items in the same cluster
    df_embeddings = df_embeddings[df_embeddings['kmeans_label']==query_label[0]]
    # convert embeddings to numpy
    np_embeddings = np.asarray(df_embeddings.iloc[:,:-2])

    # calculate manhattan distance between every item and our target item
    manhattan_dist = []
    for i in range(np_embeddings.shape[0]):
        manhattan_dist.append(np.absolute(query_embedding - np_embeddings[i,:]).sum())

    # create a dataframe with product name and manhattan distance
    df_manhattan = pd.DataFrame(data={'article_id': df_embeddings['article_id'].to_list(), 'manhattan_dist': manhattan_dist})

    # sort values ascendingly by manhattan distance
    df_manhattan = df_manhattan.sort_values(by='manhattan_dist', ascending= True)

    # print top-5 items
    for i in range(5):
        print('Top ({})\nproduct id : {}\nproduct name : {}\nproduct description : {}\n'.format(i+1, df_manhattan.iloc[i, 0], df_articles[df_articles['article_id']==df_manhattan.iloc[i, 0]].iloc[0,2], df_articles[df_articles['article_id']==df_manhattan.iloc[i, 0]].iloc[0,24]))
    print('#### search ended ####')


else :
    print('########## Guide ##########')
    print('1- python semantic.py generate-index')
    print('1.1- Generates sentence embeddings for all of the products in the database')
    print('1.2- Builds a Kmeans clustring model')
    print('1.3- Saves embeddings, cluster labels and kmeans model')
    print('2- python semantic.py cluster <product_position>')
    print('2.1- Find top-n most similar items to the selected item from the database')
    print('3- python semantic.py search <\'search sentence\'>')
    print('3.1- Converts search sentence to sentence embedding')
    print('3.2- Uses kmeans model to predict cluster label')
    print('3.3- Retrieves all items with the same cluster label')
    print('3.4- Calculates manhattan distance and return top-n most similar items')
