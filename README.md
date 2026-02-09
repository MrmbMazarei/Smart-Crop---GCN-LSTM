# Smart-Crop---GCN-LSTM
This code is a tool for crop yield forecast and onset optimization, under future climate situation.

This code is related to the following study which is under review in the Journal of Artificial Intelligence in Agriculture.

Climate-Smart Maize Cultivation: Optimizing Planting Onset with Enhanced Spatiotemporal Deep Learning Framework (2026); Mohammad M. Behbahani1*, Guiling Wang1, Emmanouil N. Anagnostou1, Fahad Khan Khadim2, Meijian Yang3,4, Amvrossios C. Bagtzoglou1*; Artificial Intelligence in Agriculture.


This script is designed to forecast crop yield across Ethiopia by learning spatial–temporal relationships between administrative units (e.g., woredas). It uses observed crop yield data together with climate drivers derived from ERA5-Land reanalysis (historical conditions) and CMIP6 climate projections (future scenarios). The workflow builds a graph of spatial units using a correlation-based adjacency matrix, then trains a GCN-LSTM (Graph Convolutional Network + LSTM) to capture both spatial connectivity and temporal dynamics. Model hyperparameters are tuned with Optuna, and the trained model is used to generate yield forecasts and evaluation metrics across cross-validation folds and multiple scenario Excel inputs.