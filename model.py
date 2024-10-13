import pandas as pd
import numpy as np
import torch
import encoder
import sys
import pickle
import helpers
from sklearn.metrics import confusion_matrix, accuracy_score


if sys.argv[1] == 'build' :
    # Loading vocabulary
    with open('vocab.pickle', 'rb') as handle:
        vocab = pickle.load(handle)
    # Loading model_metrics file
    with open('model_metrics.pickle', 'rb') as handle:
        model_metrics = pickle.load(handle)

    # Loading model params
    model_params = torch.load('model'+str(len(model_metrics['accuracy']))+'.pth', weights_only=True)

    # initializing model
    model = encoder.Encoder(len(vocab))
    # create adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # initialize loss function
    loss_function = torch.nn.CrossEntropyLoss()
    # reload model parametrs
    model.load_state_dict(model_params['model'])
    optimizer.load_state_dict(model_params['optimizer'])

    # load dataset
    X_train = pd.read_csv('../data/x_train.csv')
    X_test = pd.read_csv('../data/x_test.csv')
    y_train = pd.read_csv('../data/y_train.csv')
    y_test = pd.read_csv('../data/y_test.csv')

    # Generate dataset
    dataset = helpers.tokenize(X_train['clean_text'].to_list(), vocab)
    x_train_tensor = torch.tensor(dataset, dtype=torch.int32)
    y_train_tensor = torch.tensor(y_train['clean_label'].to_list(), dtype=torch.long)

    dataset = helpers.tokenize(X_test['clean_text'].to_list(), vocab)
    x_test_tensor = torch.tensor(dataset, dtype=torch.int32)
    y_test_tensor = torch.tensor(y_test['clean_label'].to_list(), dtype=torch.long)

    batch_size = 512
    epoch_n = int(sys.argv[2])

    for i in range(epoch_n,epoch_n+5,1) :
        print('~~~~ Starting Epoch {} ~~~~'.format(i+1))
        # Starting training phase in batches
        pointer = 0
        pointer_plus = 0
        losses = []

        print('\n#### Starting Training phase ####')
        print('-----------------------------------------')
        while pointer < x_train_tensor.shape[0] :
            print('Batch {} started ... '.format(int(pointer / batch_size) + 1))
            # prepare batch pointers
            if pointer + batch_size > x_train_tensor.shape[0] :
                pointer_plus = x_train_tensor.shape[0]
            else :
                pointer_plus = pointer + batch_size

            # pass training batch to the model
            output = model(x_train_tensor[pointer:pointer_plus, :])
            # calculate and store loss
            loss = loss_function(output, y_train_tensor[pointer:pointer_plus])
            losses.append(loss.item())
            print('loss = ', loss.item())
            # perform a backward pass to update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('updating parameters ...')
            print('Batch {} Ended '.format(int(pointer / batch_size) + 1))
            print('-----------------------------------------')
            # increament batch pointer
            pointer += batch_size
        print('#### Training phase completed ####\n\n\n')

        # starting testing phase
        pointer = 0
        pointer_plus = 0
        y_preds = []

        print('#### Starting Testing phase ####')
        print('-----------------------------------------')
        with torch.no_grad():
            while pointer < x_test_tensor.shape[0] :
                print('Batch {} started ... '.format(int(pointer / batch_size) + 1))
                # prepare batch pointers
                if pointer + batch_size > x_test_tensor.shape[0] :
                    pointer_plus = x_test_tensor.shape[0]
                else :
                    pointer_plus = pointer + batch_size

                # pass batch to model
                y_pred = model(x_test_tensor[pointer:pointer_plus, :])
                # concatenate predictions
                y_preds.append(y_pred)

                print('Batch {} Ended '.format(int(pointer / batch_size) + 1))
                print('-----------------------------------------')
                # increament batch pointer
                pointer += batch_size

            print('\ncalculating testing metrics ...')
            # concat results
            y_preds = torch.cat(y_preds, 0)

            # calculate loss
            val_loss = loss_function(y_preds, y_test_tensor).item()
            print('loss = ', val_loss)

            # calculate accuarcy
            y_pred_argmax = torch.argmax(y_preds, dim=1)
            acc = accuracy_score(y_test_tensor.detach().numpy(), y_pred_argmax.detach().numpy())
            print("Accuracy score  : {}".format(acc))

            # calculate confusion matrix
            cm = confusion_matrix(y_test_tensor.detach().numpy(), y_pred_argmax.detach().numpy())
            print('confusion matrix : ')
            print(cm)

            # calculate precision, recall and f1 scores
            cm_np = np.array(cm)
            column_sums = np.sum(cm_np, axis=0)
            row_sums = np.sum(cm_np, axis=1)
            f1_scores = []

            print('\n')
            for i in range(5):
                precision = 0
                recall = 0
                tp = cm[i,i]
                fp_tp = column_sums[i]
                fn_tp = row_sums[i]

                if fp_tp != 0 :
                    precision = tp / fp_tp
                    print('precision of class {} = {}'.format(i, tp / fp_tp))
                else :
                    print('precision of class {} = 0.0'.format(i))

                if fn_tp != 0 :
                    recall = tp / fn_tp
                    print('recall of class {} = {}'.format(i, tp / fn_tp))
                else :
                    print('recall of class {} = 0.0'.format(i))

                if precision != 0 and recall != 0 :
                    print('F1 of class {} = {}'.format(i, 2 * (precision*recall) / (precision+recall)))
                    f1_scores.append(2 * (precision*recall) / (precision+recall))
                else :
                    print('F1 of class {} = 0.0'.format(i))
                    f1_scores.append(0.0)
                print('-----------------------------------------')
            # Store model metrics of the current epoch
            model_metrics['training_loss'].append(np.array(losses).mean())
            model_metrics['validation_loss'].append(val_loss)
            model_metrics['f1'].append(f1_scores)
            model_metrics['accuracy'].append(acc)
            print('#### Testing phase completed ####\n')

    # save model metrics
    with open('model_metrics.pickle', 'wb') as handle :
        pickle.dump(model_metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # save model parameters
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, 'model'+str(len(model_metrics['accuracy']))+'.pth')
    # End code
    print('\n\nSaved progress successfully ....')



else :
    # df = pd.read_csv('../data/articles.csv')
    # print(df.info())
    # df = df[df['index_group_name'] == 'Divided']
    # select the two relevant columns and drop duplicate rows
    # df = df.iloc[:,[2,19,24]]
    # drop row with null detail_desc values
    # df = df.dropna(subset='detail_desc')
    # print(df['index_group_name'].value_counts()/df.shape[0])
    # print(df['index_group_name'].value_counts())

    # Loading model_metrics file
    # with open('model_metrics.pickle', 'rb') as handle:
    #     print(pickle.load(handle))

    # (512, 50, 128)

    x = np.random.rand(3,50,128)

    y = torch.from_numpy(x)

    print(y.shape)
    print(y.mean(dim=1).shape)
    print('here ...')
