# LP5 Deep Learning Notebook Explanations

This file explains what each notebook does, step by step, in simple words. The goal is not just to remember the code, but to understand why each choice was made.

## Big Picture

| Notebook | Problem Type | Data Type | Model | Output |
| --- | --- | --- | --- | --- |
| `1_Boston.ipynb` | Regression | Tabular housing data | Dense neural network | House price |
| `2_IMDB.ipynb` | Binary classification | Text reviews | Embedding + Dense network | Positive or negative sentiment |
| `3_disease.ipynb` | Multi-class classification | Images | CNN | Plant disease class |
| `4_stock.ipynb` | Time-series regression | Stock prices | LSTM | Next stock opening price |

The easiest way to remember the four notebooks:

- Boston: table data, predict a number.
- IMDB: text data, predict positive or negative.
- Disease: image data, predict one of many classes.
- Stock: sequence data, predict the next value.

## Common Deep Learning Flow

Most notebooks follow the same general pattern:

1. Import libraries.
2. Load the dataset.
3. Prepare the data into `X` and `y`.
4. Scale, pad, augment, or reshape the data.
5. Build the neural network.
6. Compile the model with optimizer, loss, and metrics.
7. Train the model.
8. Evaluate or visualize the result.

The difference is in how the data is prepared and which model is suitable for that data.

---

# 1. Boston Housing Notebook

File: `1_Boston.ipynb`

## What The Notebook Does

This notebook predicts house prices from tabular housing features. Each row represents one house or area, and the target column is `MEDV`, which means median house value.

This is a regression problem because the output is a number, not a class label.

## Step 1: Import Libraries

The notebook imports:

- `pandas` for reading and working with CSV data.
- `numpy` for numerical operations.
- `matplotlib` for plotting.
- `train_test_split` for splitting data into training and testing sets.
- `StandardScaler` for feature scaling.
- `Sequential` and `Dense` from Keras for building the neural network.
- `mean_squared_error` and `r2_score` for regression evaluation.

Why these choices:

- CSV data is easiest to handle with `pandas`.
- Neural networks need numerical arrays, so `numpy` helps.
- Regression needs metrics like MSE, RMSE, and R2.

## Step 2: Load Dataset

```python
data = pd.read_csv("HousingData.csv")
```

The notebook loads the Boston housing CSV file.

Why:

- The dataset is stored as rows and columns.
- `pd.read_csv()` converts it into a DataFrame that is easy to inspect and process.

## Step 3: Prepare Features And Labels

```python
X = data.drop("MEDV", axis=1)
y = data["MEDV"]
```

`X` contains all input features. `y` contains the value we want to predict.

The notebook also fills missing feature values using the column mean.

Why:

- The model cannot train properly if input values are missing.
- Replacing missing values with the column average is a simple and common fix.
- `MEDV` is removed from `X` because it is the answer, not an input.

## Step 4: Train-Test Split

The data is split into training and testing data.

Why:

- The model learns from the training set.
- The test set checks whether the model works on unseen data.
- This prevents us from judging the model only on data it already saw.

## Step 5: Feature Scaling

```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

`StandardScaler` changes features so they have mean `0` and standard deviation `1`.

Why:

- Neural networks train better when all input columns are on a similar scale.
- For example, one column may range from `0` to `1`, while another may range from `0` to `1000`.
- If we do not scale, large-value columns can dominate training.
- We use `fit_transform` only on training data to avoid data leakage.
- We use `transform` on test data so test data is scaled using training rules.

## Step 6: Build The Model

```python
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))
```

The model has:

- First hidden layer: `Dense(64, activation='relu')`
- Second hidden layer: `Dense(32, activation='relu')`
- Output layer: `Dense(1)`

Why:

- Dense layers are suitable for tabular numeric data.
- `64` and `32` neurons give the model enough capacity to learn patterns.
- `relu` helps the model learn non-linear relationships.
- The final layer has `1` neuron because we predict one number: house price.
- The final layer has no activation because regression output should be free to take any numeric value.

## Step 7: Compile The Model

```python
model.compile(
    optimizer='adam',
    loss='mean_squared_error',
    metrics=['mae']
)
```

Why:

- `adam` is a reliable default optimizer.
- `mean_squared_error` is common for regression because it penalizes large mistakes strongly.
- `mae` means mean absolute error, which is easier to understand as average prediction error.

## Step 8: Train The Model

```python
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=16,
    validation_split=0.2,
    verbose=1
)
```

Important parameters:

- `epochs=100`: the model sees the training data 100 times.
- `batch_size=16`: weights update after every 16 samples.
- `validation_split=0.2`: 20 percent of training data is used for validation.

Why:

- Boston is a small dataset, so more epochs are acceptable.
- A smaller batch size can help the model update frequently.
- Validation data helps monitor overfitting.

## Step 9: Predict And Evaluate

The model predicts prices for `X_test`.

Metrics used:

- MSE: average squared error.
- RMSE: square root of MSE, easier to understand because it is in the same unit as price.
- R2 score: tells how well the model explains the variation in the data.

Why:

- Accuracy is not suitable because this is not classification.
- Regression needs error-based metrics.

## Boston Key Memory Points

- Dataset: `HousingData.csv`
- Target: `MEDV`
- Preprocessing: fill missing values and use `StandardScaler`
- Model: Dense neural network
- Output: one numeric value
- Loss: `mean_squared_error`
- Metric: `mae`, plus MSE/RMSE/R2 after prediction
- Training: `epochs=100`, `batch_size=16`

---

# 2. IMDB Sentiment Notebook

File: `2_IMDB.ipynb`

## What The Notebook Does

This notebook predicts whether a movie review is positive or negative.

This is a binary classification problem because there are only two possible outputs:

- Negative
- Positive

## Step 1: Import Libraries

The notebook imports:

- `numpy` for array operations.
- `imdb` dataset from Keras.
- `pad_sequences` for making all reviews the same length.
- `Sequential`, `Embedding`, `Flatten`, `Dense`, and `Dropout` for the model.
- `matplotlib` for plotting accuracy.

Why:

- Text reviews have different lengths.
- Neural networks need fixed-size input.
- Embedding layers are used to learn word meaning from integer word IDs.

## Step 2: Load IMDB Dataset

```python
vocab_size = 10000
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)
```

The IMDB dataset is already tokenized. This means words are already converted into integer numbers.

Why:

- Neural networks cannot directly understand raw text.
- Integers represent words.
- `num_words=10000` keeps only the 10,000 most frequent words.
- Rare words are ignored to keep the problem smaller and simpler.

## Step 3: Pad Sequences

```python
max_length = 200
x_train = pad_sequences(x_train, maxlen=max_length)
x_test = pad_sequences(x_test, maxlen=max_length)
```

Reviews have different lengths, so the notebook makes every review length `200`.

Why:

- The model needs a fixed input shape.
- Short reviews are padded with zeros.
- Long reviews are cut down.
- `200` is a practical length that keeps enough review information without making training too slow.

## Step 4: Build The Model

```python
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=32, input_length=max_length))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
```

The model has:

- `Embedding`: converts word IDs into 32-dimensional vectors.
- `Flatten`: converts the sequence of word vectors into one long vector.
- `Dense(128, relu)`: learns patterns from the review representation.
- `Dropout(0.5)`: reduces overfitting.
- `Dense(64, relu)`: learns more compact patterns.
- `Dropout(0.5)`: again reduces overfitting.
- `Dense(1, sigmoid)`: outputs probability of positive sentiment.

Why:

- Embedding is used because word IDs need meaning.
- `relu` is used in hidden layers to learn non-linear patterns.
- `Dropout(0.5)` randomly switches off half the neurons during training, which prevents memorization.
- `sigmoid` is used because the output is binary: probability between `0` and `1`.

## Step 5: Compile The Model

```python
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```

Why:

- `binary_crossentropy` is used for two-class classification.
- `accuracy` is easy to understand: how many reviews were classified correctly.
- `adam` is a good default optimizer.

## Step 6: Train The Model

```python
history = model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.2
)
```

Important parameters:

- `epochs=5`: trains for 5 passes over the data.
- `batch_size=128`: updates after 128 reviews.
- `validation_split=0.2`: uses 20 percent of training data for validation.

Why:

- IMDB has many reviews, so fewer epochs are enough.
- A larger batch size makes training faster.
- Validation checks whether the model generalizes.

## Step 7: Evaluate

```python
loss, accuracy = model.evaluate(x_test, y_test)
```

Why:

- The test set checks performance on unseen reviews.
- Accuracy is suitable because the output is a class.

## Step 8: Predict Sample Reviews

```python
predictions = model.predict(x_test[:5])
```

The notebook checks the first few predictions.

Rule:

- If prediction is greater than `0.5`, sentiment is Positive.
- Otherwise, sentiment is Negative.

Why:

- Sigmoid gives a probability.
- `0.5` is the normal threshold for binary classification.

## Step 9: Plot Training Curves

The notebook plots:

- Training accuracy
- Validation accuracy

Why:

- If both improve together, training is healthy.
- If training accuracy increases but validation accuracy drops, the model may be overfitting.

## IMDB Key Memory Points

- Dataset: Keras IMDB dataset
- Data form: pre-tokenized integer reviews
- Preprocessing: `pad_sequences`, length `200`
- Model: Embedding + Flatten + Dense
- Embedding size: `32`
- Dropout: `0.5`
- Output activation: `sigmoid`
- Loss: `binary_crossentropy`
- Metric: `accuracy`
- Training: `epochs=5`, `batch_size=128`

---

# 3. Plant Disease Notebook

File: `3_disease.ipynb`

## What The Notebook Does

This notebook classifies plant leaf images into disease categories.

This is a multi-class classification problem because there are many possible output classes.

The notebook mentions `38` classes.

## Step 1: Import Libraries

The notebook imports:

- `numpy` for numerical operations.
- `matplotlib` for plotting accuracy and loss.
- `Sequential` for the model.
- `Conv2D`, `MaxPooling2D`, `Flatten`, `Dense`, and `Dropout` for CNN layers.
- `ImageDataGenerator` for loading and augmenting images.

Why:

- Images need convolutional layers.
- CNNs are designed to find patterns in images.
- Image generators can load images directly from folders.

## Step 2: Define Dataset Paths

```python
train_path = "Datasets/Assignment_3/train"
valid_path = "Datasets/Assignment_3/valid"
test_path = "Datasets/Assignment_3/test"
```

The images are stored in folders.

Why:

- Each class has its own folder.
- The folder name becomes the label.
- This is a common format for image classification datasets.

## Step 3: Image Preprocessing And Augmentation

The notebook uses `ImageDataGenerator`.

Training images use:

- Rescaling
- Rotation
- Zoom
- Horizontal flip

Validation and test images mainly use rescaling.

Why:

- Pixel values originally range from `0` to `255`.
- Rescaling changes them to `0` to `1`, which helps training.
- Augmentation creates slightly changed versions of images.
- This helps the model avoid memorizing the training images.
- Validation and test data should not be randomly augmented because we want honest evaluation.

## Step 4: Load Images From Folders

The notebook uses `flow_from_directory`.

Important settings:

- Image size: `128 x 128`
- `class_mode='categorical'` for training and validation
- Test data uses no labels

Why:

- CNNs need all images to have the same height and width.
- `categorical` labels are used because there are many classes.
- Test data may be used only for prediction.

## Step 5: Inspect Class Labels

The notebook prints:

- Number of classes
- Class label dictionary

Why:

- This helps us know which number corresponds to which plant disease.
- It confirms that the dataset loaded correctly.

## Step 6: Build CNN Convolutional Blocks

```python
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
```

The CNN has three convolution blocks:

- Block 1: `32` filters
- Block 2: `64` filters
- Block 3: `128` filters

Why:

- `Conv2D` finds image features like edges, textures, and patterns.
- `MaxPooling2D` reduces image size and keeps important features.
- `Dropout(0.25)` reduces overfitting.
- Filters increase from `32` to `64` to `128` because deeper layers learn more complex features.

## Step 7: Build Classifier Head

```python
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(train_data.num_classes, activation='softmax'))
```

Why:

- `Flatten` converts image feature maps into a single vector.
- `Dense(128)` combines learned image features.
- `Dropout(0.5)` strongly reduces overfitting before the final prediction.
- Final Dense layer has one neuron per class.
- `softmax` gives a probability distribution across all classes.

## Step 8: Compile The Model

```python
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

Why:

- `categorical_crossentropy` is used for multi-class classification.
- `accuracy` tells how many images were classified correctly.
- `adam` is a reliable optimizer.

## Step 9: Train

```python
history = model.fit(
    train_data,
    validation_data=valid_data,
    epochs=10
)
```

Why:

- The model trains on images from `train_data`.
- It validates using a separate validation folder.
- `epochs=10` gives the CNN multiple chances to learn visual patterns.

## Step 10: Save Model

```python
model.save("plant_disease_cnn_model.keras")
```

Why:

- Saving the model allows it to be reused later without training again.
- `.keras` is the modern Keras model format.

## Step 11: Print Final Accuracy

The notebook prints final training and validation accuracy.

Why:

- It gives a quick final summary of model performance.
- Comparing training and validation accuracy helps detect overfitting.

## Step 12: Predict Test Images

The notebook predicts classes for test images.

It uses:

```python
np.argmax(predictions, axis=1)
```

Why:

- Softmax gives one probability per class.
- `argmax` chooses the class with the highest probability.

## Step 13: Plot Accuracy And Loss

The notebook plots:

- Training accuracy
- Validation accuracy
- Training loss
- Validation loss

Why:

- Accuracy shows how often predictions are correct.
- Loss shows how much the model is still making mistakes.
- If validation loss increases while training loss decreases, overfitting may be happening.

## Disease Key Memory Points

- Dataset: image folders under `Datasets/Assignment_3`
- Classes: `38`
- Preprocessing: rescale and augment images
- Image size: `128 x 128`
- Model: CNN
- Conv filters: `32`, `64`, `128`
- Dropout: `0.25` in CNN blocks, `0.5` before output
- Output activation: `softmax`
- Loss: `categorical_crossentropy`
- Metric: `accuracy`
- Training: `epochs=10`, validation folder
- Special step: saves model as `.keras`

---

# 4. Google Stock Notebook

File: `4_stock.ipynb`

## What The Notebook Does

This notebook predicts the next Google stock opening price using previous stock prices.

This is a time-series regression problem.

It is regression because the output is a number. It is time-series because order matters.

## Step 1: Import Libraries

The notebook imports:

- `numpy` for arrays.
- `pandas` for CSV data.
- `matplotlib` for plotting real and predicted prices.
- `MinMaxScaler` for scaling prices.
- `Sequential`, `LSTM`, `Dense`, and `Dropout` for the model.

Why:

- Stock data is stored in a CSV.
- LSTM is useful for sequence data.
- MinMax scaling is common for stock prices.

## Step 2: Load Dataset

```python
data = pd.read_csv("Datasets/google.csv")
```

The notebook loads Google stock price data.

It predicts the `Open` price.

Why:

- The task focuses on one column to keep the model simple.
- Predicting one numeric column makes it univariate time-series regression.

## Step 3: Select Column, Split, And Scale

```python
training_data = data[["Open"]]
```

The notebook keeps only the `Open` column.

It splits by index, not randomly.

Why:

- In time-series data, order matters.
- Random split would mix future and past data, which can cause data leakage.
- The model should train on earlier prices and test on later prices.

The notebook uses:

```python
MinMaxScaler()
```

Why:

- Stock prices can have different ranges.
- LSTMs usually train better when values are between `0` and `1`.
- MinMax scaling keeps the shape of the price movement while shrinking the range.

## Step 4: Create Sliding Windows

The notebook creates input windows of `5` days.

Meaning:

- Input: previous 5 days of `Open` prices.
- Output: next day's `Open` price.

Why:

- LSTM needs a sequence.
- A sliding window converts one long price column into many short training examples.
- The model learns from recent price movement.

## Step 5: Reshape For LSTM

```python
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
```

LSTM expects input shape:

```text
samples, timesteps, features
```

For this notebook:

- Samples: number of windows
- Timesteps: `5`
- Features: `1`, because only `Open` price is used

Why:

- Dense models use flat input.
- LSTM models need a time dimension.

## Step 6: Build LSTM Model

```python
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50))
model.add(Dropout(0.2))

model.add(Dense(units=1))
```

The model has:

- Five LSTM layers
- `50` units in each LSTM layer
- `Dropout(0.2)` after each LSTM
- Final `Dense(1)` output

Why:

- LSTM layers remember patterns across time.
- `units=50` gives each LSTM enough memory capacity.
- `return_sequences=True` is used for the first four LSTMs so the next LSTM receives a full sequence.
- The last LSTM does not use `return_sequences=True` because it only needs to send the final learned summary to the Dense layer.
- `Dropout(0.2)` reduces overfitting.
- `Dense(1)` predicts one number: the next scaled stock price.

## Step 7: Compile The Model

```python
model.compile(optimizer='adam', loss='mean_squared_error')
```

Why:

- This is regression, so `mean_squared_error` is suitable.
- No accuracy metric is used because predicting a stock price is not classification.

## Step 8: Train

```python
history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=4,
    verbose=1
)
```

Important parameters:

- `epochs=10`
- `batch_size=4`

Why:

- Time-series windows are smaller and sequential.
- A small batch size updates the model frequently.
- `10` epochs gives the LSTM repeated chances to learn patterns.

## Step 9: Prepare Test Data And Predict

The notebook includes the 5 days before the test period.

Why:

- To predict the first test day, the model needs the previous 5 days.
- Without those previous days, the first test input window cannot be created.

The predictions are inverse transformed after prediction.

Why:

- The model predicts scaled values.
- Inverse scaling converts predictions back to real stock prices.

## Step 10: Visualise Results

The notebook plots:

- Real stock price
- Predicted stock price

Why:

- A line plot makes it easy to see whether the prediction follows the real trend.
- For time-series prediction, the shape of the curve matters.

## Stock Key Memory Points

- Dataset: `Datasets/google.csv`
- Target column: `Open`
- Problem: time-series regression
- Preprocessing: `MinMaxScaler`
- Window size: previous `5` days
- LSTM input shape: samples, timesteps, features
- Model: five `LSTM(50)` layers
- Dropout: `0.2`
- Output: `Dense(1)`
- Loss: `mean_squared_error`
- Training: `epochs=10`, `batch_size=4`
- Special rule: split by time order, not randomly

---

# Final Comparison

## Dataset Difference

| Notebook | Dataset | Input Shape Idea |
| --- | --- | --- |
| Boston | Housing CSV | Rows and numeric columns |
| IMDB | Built-in review dataset | Padded word index sequences |
| Disease | Image folders | 128 x 128 RGB images |
| Stock | Google CSV | 5-day price windows |

## Model Difference

| Notebook | Model Type | Why |
| --- | --- | --- |
| Boston | Dense network | Works for tabular numeric data |
| IMDB | Embedding + Dense | Learns from word IDs |
| Disease | CNN | Finds visual patterns in images |
| Stock | LSTM | Learns patterns over time |

## Output Difference

| Notebook | Output Layer | Meaning |
| --- | --- | --- |
| Boston | `Dense(1)` | One price value |
| IMDB | `Dense(1, sigmoid)` | Probability of positive review |
| Disease | `Dense(num_classes, softmax)` | Probability for each disease class |
| Stock | `Dense(1)` | One stock price value |

## Loss Difference

| Notebook | Loss | Why |
| --- | --- | --- |
| Boston | `mean_squared_error` | Regression |
| IMDB | `binary_crossentropy` | Two-class classification |
| Disease | `categorical_crossentropy` | Multi-class classification |
| Stock | `mean_squared_error` | Regression |

## Preprocessing Difference

| Notebook | Preprocessing | Why |
| --- | --- | --- |
| Boston | Fill missing values, `StandardScaler` | Clean and scale tabular features |
| IMDB | `pad_sequences` | Make reviews equal length |
| Disease | Rescale and augment images | Normalize pixels and reduce overfitting |
| Stock | `MinMaxScaler`, sliding windows | Scale prices and create sequences |

## One-Line Memory Hooks

- Boston: StandardScaler plus Dense regression for `MEDV`.
- IMDB: Pad reviews, use Embedding, sigmoid sentiment output.
- Disease: ImageDataGenerator plus CNN plus softmax over 38 classes.
- Stock: MinMaxScaler plus 5-day windows plus stacked LSTM.

