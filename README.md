# H&M-semantic-search
<br>
Goal of this project:<br>
Implementing semantic search using data from the "H&M Personalized Fashion Recommendations" on <a href="https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations" target="blank">Kaggle</a>. In order to do so, we build a nlp classifer using the transformer encoder. Then, we use word embeddings from our model to generate sentence embeddings and perform semantic search
<br><br>
Please download the dataset from <a href="https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations" target="blank">Kaggle</a> before running the project source code. Unzip and place all csv files in a folder named "../data". The data was included in this repository due to its large size
<br><br>
Semantic search can be used to retrieve items/documents from a large database that have similar semantic meaning (similar sentence embeddings) to a search query. Let us take YOUTUBE as an example, when a user types a query in the search bar, we might use text query and find videos with descriptions that have similar semantic meaning (similar sentence embeddings). Another good example would be LinkedIn. We can use the description of a specific job post, find applicants/users that have headlines/bios with similar semantic meaning (similar sentence embeddings), finally recommend those users and vice versa.
<br><br>

## project files
1- helpers.py :
<br>Contains commonly used code snippets
<br>Used to clean, standardize and Tokenize text sentences
<br><br>
```
import helpers
```
<br>
2- preprocess.py :
<br>Generate vocabulary, initialize and save classifier model (option = 'init')
<br>Generate training and testing datasets (option = 'train-test')
<br><br>

```
python preprocess.py [option]
```
<br>
3- model.py :
<br>Use training data to train our model
<br>Use testing data to evaluate our model
<br>Save model parametrs
<br><br>

```
python model.py build
```
<br>
4- semantic.py :
<br>Generate and save sentence embeddings for every item in the dataset (option = 'generate')
<br>Test semantic search on random items in the dataset (option = 'cluster') (option2 = <item_position:int>)
<br>Test semantic search on random text queries (option = 'search') (option2 = <query:str>)
<br><br>

```
python semantic.py [option] [option2]
```
<br>
5- encoder.py :
<br>encoder model used to build our classifier and generate sentence embeddings
<br><br>

```
import encoder
```
<br>
6- vocab.pickle :
<br>Dictionary that maps all vocabulary words to unique numeric tokens
<br>saved in a pickle file
<br><br>
7- model.pth :
<br>Contains parameters of our final model and optimizer
<br><br>
8- model_metrics.pickle :
<br>Contains training loss, validation loss, accuracy and f1 scores of each epoch during our training/testing phase
<br><br>

## Installations
In order to clone the project files and run them on your machine, a number of packages and libraries must be installed
<br><br>
**1- python 3.10**
<br><br>
**2- Pandas**
<br>
  To install
<br>
```
# conda
conda install -c conda-forge pandas
# or PyPI
pip install pandas
```
<br>

**3- Numpy**
<br>
  To install
<br>
```
# conda
conda install -c anaconda numpy
# or PyPI
pip install numpy
```
<br>

**4- scikit-learn**
<br>
  To install
<br>
```
# conda
conda create -n sklearn-env -c conda-forge scikit-learn
# or PyPI
pip install -U scikit-learn
```
<br>

**5- torch**
<br>
  To install
<br>
```
# conda
conda install pytorch torchvision torchaudio cpuonly -c pytorch
# or PyPI
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```
<br>

**6- nltk**
<br>
  To install
<br>
```
# conda
conda install nltk
# or PyPI
pip install nltk
```
<br>
<br>

## Sources
<a href="https://pytorch.org/get-started/locally/">Installing torch</a>
<br>
