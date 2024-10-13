import torch
import pandas as pd
import numpy as np
import helpers
import pickle
import encoder

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
    pd.concat([final_df.reset_index(drop=True), df.iloc[:,0].reset_index(drop=True)], axis=1).to_csv('../data/embeddings.csv')
    print('Saved embeddings successfully ...')
