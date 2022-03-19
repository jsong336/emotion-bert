ENVIRONMENT=tf-gpu-nlp
conda create -n $ENVIRONMENT python=3.9 && \\
conda activate $ENVIRONMENT && \\
conda install -y -c anaconda tensorflow
conda install -y -c anaconda tensorflow-gpu && \\
conda install -y -c conda-forge transformers && \\
conda install -y -c conda-forge/label/cf202003 transformers && \\ 
conda install -c anaconda scikit-learn && \\
conda install -c anaconda jupyter && \\ 
pip install -y --upgrade tensorflow-hub && \\
pip install -y pandas && \
pip install -y tweet-preprocessor && \\ 
pip install -q emoji
