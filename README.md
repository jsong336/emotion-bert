# go-emotions

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
