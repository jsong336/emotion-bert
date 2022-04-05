# Finding Human Emotions from Text Comments

__CS4442B Artificial Intelligence 2 Final Project (Undergraduate)__

Jeongwon Song: jsong336@uwo.ca <br/>
Jason Koo: jkoo26@uwo.ca <br/>

Fine-grained emotion detectiong using BERT and GoEmotions dataset: https://arxiv.org/abs/2005.00547. Details in <a href="/report.pdf" target="_blank" > report.pdf </a>.

### Setup with Google Colab

The project was developed in Google Colab utilizing Google Drive as storage. To setup the running environment please follows below steps. 

1. Create a colab notebook and clone the project. 

```python3
from google.colab import drive
drive.mount('/content/drive/')

repos_dir = '/content/drive/MyDrive/{where you want to put in google drive}'
repos = 'fine-grained-emotions-bert' # our repository name
url = "https://github.com/jsong336/fine-grained-emotions-bert.git"

%cd $repos_dir
! git clone $url
%cd $repos_dir/$repos
! git pull 
```
2. Go to Kaggle and create developer API credential. 
```python3
import os

os.environ['KAGGLE_USERNAME'] = ""
os.environ['KAGGLE_KEY'] = ""

def download_from_kaggle(url, target_dir):
  dataset_name = url.split('/', 1)[-1]
  dirname = os.path.join(target_dir, dataset_name)
  ! mkdir $dirname
  ! kaggle datasets download -d $url
  ! unzip $dataset_name -d $dirname
  return 
  
input_dir = repos_dir + '/inputs'
download_from_kaggle('shivamb/go-emotions-google-emotions-dataset', input_dir)
download_from_kaggle('ishivinal/contractions', input_dir)
download_from_kaggle('bittlingmayer/spelling', input_dir)
```

Go to your Google Drive and make sure you have the repository cloned and datasets downloaded

3. Go to `notebooks/` and run `main_prepare_dataset.ipynb` and you should have train & test datasets splitted in the `inputs/`

4. Run `main_bert_gru.ipynb` and `main_bert_dense.ipynb` notebooks to train the models. (Careful, these notebooks create a checkpoint on your Google Drive and easily take up a lot of Google Drive space)

5. Run `view_model_analysis.ipynb` to compare the models. 

Because all of our codes were developed in google colab and archives were stored in google drive, any `view_*` or `main_*` codes contains following block

```python3
import os 

# os.environ['GO_EMOTIONS_COLAB_WORKDIR'] = '/content/drive/MyDrive/Notebooks/Repository/go-emotions/notebooks'
colab_workdir = os.environ.get('GO_EMOTIONS_COLAB_WORKDIR')

if colab_workdir:
    print('Running with colab')
    from google.colab import drive
    drive.mount('/content/drive')
    %cd $colab_workdir
    !pip install -q -r ../requirements.txt
else:
    print('Running with jupyter notebook')
```
You might need to update `os.environ['GO_EMOTIONS_COLAB_WORKDIR'] = {cloned work directory in google drive}`.
