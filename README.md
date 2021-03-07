# Template for deeplearning project

### Folder Structure
```python
assets/          # materials used in document or for test
configs/         # configure file for training
results/         # the output of test or inference
scripts/         # scripts used to preprocess data and inference
src/
  criterions/    # loss and metric function
  datasets/      # pytorch dataset and dataloader
  models/        # network model and components (both import in train and inference)
  trainers/      # the training process for the project (only incude code for training)
  optimizers/    # customized optimizers(optimizers provided by pytorch are included by default)
  schedulers/    # customized schedulers(schedulers provided by pytorch are included by default)
  utils/
tests/           # code to test datasets, models or other parts
third_party/     # third party code (should be self-contained)
train.py         # general entrance for training
```

### Training Configuration
The configuration is powered by [mlconfig](https://github.com/narumiruna/mlconfig) and wrote in `yaml` standard. There is the simple example about how to configure the model. To initialize the model, we should first add the `models` folder into the mlconfig search space.

```python
# In models/my_model.py
import mlconfig

@mlconfig.register()
class MyModel(nn.Module):
  ...

# In train.py
import models.my_model

# train.yaml
#
# model:
#   name: MyModel
#   in_nc: 3
#   n_classes: 1024
#   nd: 64
config = mlconfig.load("train.yaml")
config_model = config.model

# This line includes two things
# 1. find the function or class named `MyModel`
# 2. call the function or initialize the class with arguments defined in yaml or passed by key-value pairs

model = config_model(other_key=other_value)

```

### Logging
Use [MLflow](https://mlflow.org) as training logger
