ENVIRONMENT=go-emotions
conda create -n $ENVIRONMENT python=3.9
conda activate $ENVIRONMENT
pip install -U -r requirements.txt