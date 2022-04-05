# Finding Human Emotions from Text Comments

__CS4442B Artificial Intelligence 2 Final Project (Undergraduate)__

Jeongwon Song: jsong336@uwo.ca <br/>
Jason Koo: jkoo26@uwo.ca <br/>

Before running, please download following files to `inputs/`

```python3
def download_from_kaggle(url, target_dir=input_dir):
  dataset_name = url.split('/', 1)[-1]
  dirname = os.path.join(target_dir, dataset_name)
  ! mkdir $dirname
  ! kaggle datasets download -d $url
  ! unzip $dataset_name -d $dirname
  return 
  
download_from_kaggle('shivamb/go-emotions-google-emotions-dataset')
download_from_kaggle('ishivinal/contractions')
download_from_kaggle('bittlingmayer/spelling')
```

Once you have all csv files, 

1. Go to `notebooks/` folder which contains all of our source code & notebooks 
2. Run `main_prepare_dataset.ipynb`
3. Check if you have train and test csv datasets in `outputs/` folder. 
4. Then you could run `main_bert_gru.ipynb` and `main_bert_gru.ipynb` to train the models. 
5. Once models are trained, you could use `view_model_analysis.ipynb` to compare the models. 

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

If you wish to run these notebooks, then you might need to update `os.environ['GO_EMOTIONS_COLAB_WORKDIR'] = {workdir}` or `export GO_EMOTIONS_COLAB_WORKDIR={workdir}`.
