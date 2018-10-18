#PhysioRNN

An RNN model for mortality prediction of ICU patients, with data from the Physionet 2012 competition. The architecture handles multivariate time series with missing values that incorporates representations of the patterns of missingness. 
Makes use of a missing data indicator as described by Lipton et al. (2016), augmenting the time series input with missing feature indicators, based on assumptions about the recording of clinical features:
Clinical variables are recorded based on their significance  to the patient’s state, as deemed necessary by the physicians. Therefore, indicator variables exploiting this informative missingness could incorporate deeper patterns, such as the abnormality of a specific measurement (physicians may not choose to measure a value if they assumed it to be typical), or in the combination of measurements, which may point to a larger condition.


### To train a model on set A
python3 train.py -opt momentum --name name_of_model

### To evaluate a model on test set C
python3 evaluate.py -r 'path_to_ckpt'

### To evaluate on pretrained best-fit model
python3 evaluate.py -r 'path_to_ckpt' 'checkpoints/best/physiornn_best/rnn_physiornn_112_epoch80.ckpt-80'
