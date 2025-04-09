large brain model

classification of states, alert, transition, drowsy

install requirements.txt

run training/preprocess.py to label dataset, ICA, filtering ---> saves to processed_data_ica

run training/model.py to train vq vae and transformer model
    optionally use args --load-vq and/or --load-transformer to skip to metrics

