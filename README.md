large brain model

classification of states, alert, transition, drowsy

run training/preprocess.py to label dataset, ICA, filtering ---> saves to processed_data_ica

run training/model.py to train vq vae and transformer model
use args --load-vq and/or --load-transformer to skip to metrics