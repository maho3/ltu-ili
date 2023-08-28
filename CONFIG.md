Configuration
---------------------------------
This document serves as a reference for the functionality of ltu-ili. 

There are three stages to the inference pipeline which can be independently configured:
- **Data Loading**: Loading various structured data into memory
- **Training**: Training neural networks from the loaded data and saving them to file.
- **Validation**: Loading neural networks from file, sampling posteriors on the test set, and computing metrics.

In the following examples, we will provide an example configuration, followed by a list of the available options.

## Data Loading
Example for loading Pk from Quijote simulations:
