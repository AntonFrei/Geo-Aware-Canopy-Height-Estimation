# Evaluating the Impact of Geocoordinate Integration on Canopy Height Estimation Accuracy
[Adapted from https://github.com/AI4Forest/Global-Canopy-Height-Map ; https://arxiv.org/abs/2406.01076]

https://github.com/AntonFrei/Geo-Aware-Canopy-Height-Estimation.git

## Modified:

_training_

&nbsp;&nbsp;&nbsp;&nbsp;- main.py 

```Adapted hyperparameters; use of standardization; addition of: use_coord_encoding, coord_encoder, coord_injection_mode```

&nbsp;&nbsp;&nbsp;&nbsp;- runner.py 

```init added: use_coord_encoding, coord_encoder, coord_incection_mode; deactivation of fixval; additional function: collate_fn, EarlyStopping; get_model: dynamic adaption of in_channels, CoordInjectionModelWrapper for addition of coordinates before final 3x3 convolution;  eval & train: adds encoded coordinates to batch data, include early stopping (train)```

&nbsp;&nbsp;&nbsp;&nbsp;- config.py 

```updated means, stds and percentiles based on compute_dataset scripts; PreprocessedSatelliteDataset: addition of use_coord_encoding, coord_encoder, coord_injection_mode to init, update of getitem to create encoded coordinate vectors (if use_coord_encoding) and add them to the input tensor (if coord_injection_mode == input) or return them (if coord_injection_mode == feature_maps)```


## Added:

_training_

&nbsp;&nbsp;&nbsp;&nbsp;- encoder.py  

```contains different encoding functions and the vector size of each embedding```

&nbsp;&nbsp;&nbsp;&nbsp;- coord_injection_model.py  

```search for final convolution layer in segmentation head and replace with the generated maps of the embedding vector```

&nbsp;&nbsp;&nbsp;&nbsp;- generate_satclip_embeddings.py  

```Generates a csv with each location (train, test and val) and its satclip embedding, used in the satclip encoder function, as a simple hashlookup is simpler than generating it freshly from each new location, would need to adapted if changed into "production" ```

_testing_

&nbsp;&nbsp;&nbsp;&nbsp;- testing.py  

```uses a previously trained model for testing```

&nbsp;&nbsp;&nbsp;&nbsp;- bootstrap_eval.py

&nbsp;&nbsp;&nbsp;&nbsp;- download_model.py
