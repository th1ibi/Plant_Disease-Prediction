# Plant_Disease-Prediction
Dataset's link : https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset

**Transfer learning** is a machine learning technique where a model developed for a particular task is reused as the starting point for a model on a second task. This approach is particularly useful when the second task has limited labeled data, allowing the model to benefit from the knowledge gained from a different, but related, task.
<div align="center">
    <a><img width="600" src="images/transfer learning.jpg" alt="soft"></a>
</div>

**Available models**  : https://keras.io/api/applications/


<div align="center">
    <a><img width="400" src="images/available models.jpg" alt="soft"></a>
</div>


### 1. **Base Model Creation (VGG16)**
   - **Load VGG16**: The pre-trained VGG16 model is loaded without its top layer (`include_top=False`). This model is trained on the ImageNet dataset and is repurposed here for plant disease classification.
   - **Freeze Layers**: All layers in the VGG16 model are frozen to retain their pre-trained weights during training.

### 2. **Model Construction**
   - **Adding layers**: The base model is extended with a flattening layer and a dense (fully connected) output layer with 38 units (one for each class) and a softmax activation function for classification.

### 3. **Model Compilation and Training**
   - **Compile model**: The model is compiled using the Adam optimizer and sparse categorical cross-entropy loss, which is suitable for multi-class classification problems.
   - **Train the model**: The model is trained for 20 epochs on the training dataset and validated on the validation dataset. The history of accuracy and loss is stored for later visualization.

***
<div align="center">
    <a><img width="800" src="images/fine tuning.jpg" alt="soft"></a>
</div>

***

### 4. **Fine-Tuning**
   - **Unfreeze base model**: The base model layers are unfrozen to allow fine-tuning, adjusting the pre-trained weights to improve performance on the new dataset.
   - **Early stopping**: An early stopping callback is added to prevent overfitting by stopping training if the validation performance doesn’t improve for five consecutive epochs.

### 5. **Fine-Tuning Model Training**
   - **Train again**: The model is retrained with the base model layers unfrozen, allowing for fine-tuning.
