import tensorflow as tf
import numpy as np
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly


# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_visible_devices(devices=gpus[0], device_type='GPU')
tf.config.experimental.set_memory_growth(device=gpus[0], enable=True)

data = []
labels = []
"""
# Predicting the performed action

# This dataset consists of device motion data. 
### For more info: https://www.kaggle.com/malekzadeh/motionsense-dataset

# The goal :dart:
Given a sequence of signals determine what type of action the subject is performing: 
- Going downstairs
- Going upstairs
- Sitting
- Standing
- Walking
- Jogging

"""
# def unzip_data(path, output):
#     zipF = zipfile.ZipFile(open(path, "rb"))
#     zipF.extractall(output)
#     zipF.close()



# def load_data(path):
    
#     print("Loading data...")

#     for root, dirname, filename in os.walk(path):
#         for dir_ in dirname:  # explore each folder within A_DeviceMotion_data
#             filenames = os.listdir(os.path.join(root, dir_))
#             for file in filenames:  # explore each file within the folder
#                 class_name = dir_[:3]  # create a label based on the folder name: dws, jog, sit, std, ups, wlk
#                 with open(os.path.join(root, dir_, file), "r") as csv_file:
#                     next(csv_file)  # skip the header
#                     for row in csv_file:
#                         row_as_flow = np.fromstring(row[:-2], np.float, sep=',')[1:]  # convert to float, remove col id
#                         data.append(row_as_flow)  
#                         labels.append("{}".format(class_name))
    
#     print("Data loading done...")
#     return data, labels


# def save_to_csv(data, labels):
    
#     print("Saving to csv...")
#     data = np.array(data)
#     labels = np.array(labels)
#     labels = np.expand_dims(labels, axis=-1)

#     concatenated = np.concatenate((data, labels), axis=1)
#     print(concatenated.shape)

#     column_names = ["attitude.roll", "attitude.pitch", "attitude.yaw", "gravity.x", "gravity.y", "gravity.z", "rotationRate.x", "rotationRate.y", "rotationRate.z", "userAcceleration.x", "userAcceleration.y", "userAcceleration.z", 'labels']

#     df = pd.DataFrame(concatenated, columns=column_names)
#     print(df.head())
#     df.to_csv("./pycharm/tmp/activity_timeseries/data.csv", index=False)
#     print("Saving to csv done...")

# # not needed anymore
# if not os.path.exists("./pycharm/tmp/activity_timeseries/data.csv") and False:
#     unzip_data("./pycharm/tmp/activity_timeseries/archive.zip", "./pycharm/tmp/activity_timeseries/")

# # not needed anymore
# if not os.path.exists("./pycharm/tmp/activity_timeseries/data.csv") and False:
#     data, labels = load_data("./pycharm/tmp/activity_timeseries/A_DeviceMotion_data/A_DeviceMotion_data/")
#     save_to_csv(data, labels)


"""
# Exploring the data :page_with_curl:
"""

@st.cache
def load_csv(path):
    data = pd.read_csv(path, header=0, index_col=False, error_bad_lines=False)
    return data

url = "https://www.dropbox.com/s/pcqz4qidzie50xb/data.csv?dl=1"
data = load_csv(url)

st.dataframe(data.loc[:10], width=1900)
st.text("Shape of the data (including the labels): " + str(data.shape))

print(data.shape)

def uniques(data):
    return np.unique(data['labels'])


def plot_class_distribution(data):
    st.write("""
    ## Class sample distribution
    """)
    distribution = {}
    counts = []
    uniques_ = uniques(data)
    for unique in uniques_:
        counts.append(len(data[data.labels==unique]))
    
    distribution['labels'] = uniques_
    distribution['sample count'] = counts
    df = pd.DataFrame.from_dict(distribution)
    fig = px.bar(distribution, x='labels', y='sample count' )
    st.plotly_chart(fig)


def plot_features(data, trends=True):
    labels = uniques(data)
    
    key='Trends'
    if not trends:
        key='Distributions'

    st.write("""
    ##""", key, """ per class
    As seen above, each class/label has different number of samples, to display the""", key.lower(), """of the 12 features please select the number of samples to show.

    """)
    
    selection = st.selectbox("Select a class", ['downstairs', 'jog', 'sit', 'stand', 'upstairs', 'walk'], key=key)
    parse_selection = {
        "downstairs": "dws",
        "jog": "jog",
        "sit": "sit",
        "stand": "std",
        "upstairs": "ups",
        "walk": "wlk"
    }
    """

    """
    # to display sliders
    slider = None
    sampled_data = None
    for label in labels:
        if label == parse_selection[selection]:
            sampled_data = data[data.labels==label].reset_index()  # reset indices
            slider = st.slider(
                "Samples to display for {}".format(label), 
                1, 
                len(sampled_data), 
                int(len(sampled_data)*.2),
                1,
                key=key
            )
        
    fig = plotly.subplots.make_subplots(rows=3, cols=4)
    positions = [
        (1,1), (1,2), (1,3), (1,4),
        (2,1), (2,2), (2,3), (2,4),
        (3,1), (3,2), (3,3), (3,4),
    ]
    
    st.text("Hint: There is a maximization botton at the top right corner of the figure, try it!")
    
    # to display the figures iteratively
    if trends:
        for column, position in zip(data.columns[:-1], positions):
            fig.add_trace(
                go.Scatter(
                    x=np.array(range(len(sampled_data[:slider]))), 
                    y=sampled_data[column].loc[:slider],
                    name=column
                ),
                row=position[0], col=position[1]
            )

    else:
        for column, position in zip(data.columns[:-1], positions):
            fig.add_trace(
                go.Histogram(
                    # x=np.array(range(len(sampled_data[:slider]))), 
                    x=sampled_data[column].loc[:slider],
                    # histnorm='probability'
                    name=column
                ),
                row=position[0], col=position[1]
            )

    st.plotly_chart(fig)
    


def exploratory_analysis(data):
    # class distribution
    plot_class_distribution(data)
    # trends per variable
    plot_features(data)
    # variable distribution
    plot_features(data, False)
    

"""
### Because we want to give the user a way to reduce the number of samples we need to do so by preserving the time dependency, that is, we can't sample the dataset as usual.

## Strategy:
- Control the number of samples **per class**.
- If they *reduce* the number of samples, we will cut the samples per class ie: ```class_samples[:num_samples]```
- If they *increase* the number of samples, we will allow the samples per class ie: ```class_samples[:num_samples]```
- If they set ```num_samples``` to a number greater than the actual number of samples per class we cap the value to ```len(class_samples)```.

## Another strategy (not implemented):
- Frequency conversion and resampling of time series ie: skip every other sample.
"""
len_max_sampled_class = len(data[data.labels=='wlk'])
samples = st.number_input(
    "Select samples to work with (default length of majority class): ", 
    10000, 
    len_max_sampled_class, 
    len_max_sampled_class, 
    10000)
data = data.groupby('labels', group_keys=False).apply(lambda x: x.take(list(range(min(samples, len(x))))))

print(data.describe())
print(len(data))
print(data.shape)

exploratory_analysis(data)

"""
# Processing the data :hammer:

## We create a Tensorflow Dataset:
- We make sure we split the data with training/testing ratio.
- We create windows with X ammount of time steps and drop the reminder samples.
- We transform our labels into one-hot vectors (given that the labels are strings)
- We shuffle the data (the windows in this case)
- We create training and validating batches of Y ammount of windows.

"""
# le = preprocessing.LabelEncoder()  # transform string classes to numeric
# le.fit(np.unique(data['labels']))
# unique_labels = len(le.classes_)
# unique_labels

# print(labels) 

@st.cache
def labels_to_numeric(labels):
    # dictionary to translate from string to number
    unique_labels = np.unique(labels)
    labels_dict = dict(enumerate(unique_labels))
    unique_labels_dict = {}
    for key, value in labels_dict.items():
        unique_labels_dict[value] = int(key)

    lbls = []
    for sample in labels.to_numpy():
        lbls.append(unique_labels_dict[sample[0]])

    # labels are now numbers from 0 to 5
    return lbls


# @st.cache
def prepare_data(data, labels, train_split=0.7, window_size=500, batch_size=100):
    training_chunk = int(data.shape[0]*train_split)
    print("Training chunk: ", training_chunk)
    X = tf.data.Dataset.from_tensor_slices(data)
    X = X.window(size=window_size, shift=1, drop_remainder=True)
    X = X.flat_map(lambda x: x.batch(window_size))

    y = tf.data.Dataset.from_tensor_slices(labels)
    y = y.window(size=window_size, shift=1, drop_remainder=True)
    y = y.flat_map(lambda x: x.batch(window_size))
    y = y.map(lambda x: tf.one_hot(x, len(np.unique(labels))))

    dataset = tf.data.Dataset.zip((X, y))
    dataset = dataset.shuffle(10000, seed=42)


    train, test = dataset.take(training_chunk), dataset.skip(training_chunk)

    train = train.batch(batch_size).prefetch(1)  # ~11300 batches with batch size of 100 and split 0.8
    test = test.batch(batch_size).prefetch(1)
    batches = int(training_chunk / batch_size)
    print(train, test)
    return train, test, batches


"""
### Please select values for the data preparation. The default values are sensible values defined by the rounds of exploratory analysis.
Note that the more samples you specify, the *more time* it will take for both processing the data and training the models.

"""
# samples = st.number_input("Select samples to work with: ", 10000, len(labels), 300000, 10000)
window_size = st.number_input("Select the window size: ", 10, 1000, 500, 10)
batch_size = st.number_input("Select the batch size: ", 10, 200, 100, 10)
train_split = st.number_input("Select the train split: ", 0.1, 0.9, 0.7, 0.1)

X, labels = data[["attitude.roll", "attitude.pitch", "attitude.yaw", "gravity.x", "gravity.y", "gravity.z", "rotationRate.x", "rotationRate.y", "rotationRate.z", "userAcceleration.x", "userAcceleration.y", "userAcceleration.z"]], data[['labels']]

print(X.shape)
print(labels.shape)

# labels = le.transform(labels)

@st.cache(suppress_st_warning=True)
def checkpoint():
    prep_data = st.button('Prepare data')
    if not prep_data:
        st.warning('Please prepare the data before proceeding.')
        st.stop()
    st.success('Thank you for preparing the data')

checkpoint()


labels = labels_to_numeric(labels)
train, test, batches = prepare_data(X, labels, train_split, window_size, batch_size)

st.write(
    """Given the """, X.shape[0], """samples, the number of batches for the training set would be ~""", batches
)

"""
## Training data:
"""
train
"""
## Testing data:
"""
test

"""
The shape of the data corresponds to (batch_size, time_steps, features_dimension), in this case None means that the model can take on any number of samples per batch and any number of time steps. As the data and the labels were treated separately and then zipped, we get two of these, one for the data features and one for the labels. Note that the labels now have 6 dimensions as they are now one-hot vectors.
"""

"""
# Defining the model architecture :star2:
Select as many as LSTM and Dense layers as you wish to add to the model architecture.
Play with the values and see the training results. Which settings were the best for you?
"""
"""
## LSTM layers
"""
bi_layers = st.slider("LSTM Layers", 1, 3, 1, 1)
if bi_layers == 1:
    lstm_units_01 = st.number_input("LSTM units for first LSTM", 1, 40, 40, 5)
    lstm = [lstm_units_01]
elif bi_layers == 2:
    lstm_units_01 = st.number_input("LSTM units for first LSTM", 1, 40, 40, 5)
    lstm_units_02 = st.number_input("LSTM units for second LSTM", 1, 40, 40, 5)
    lstm = [lstm_units_01, lstm_units_02]
elif bi_layers == 3:
    lstm_units_01 = st.number_input("LSTM units for first LSTM", 1, 40, 40, 5)
    lstm_units_02 = st.number_input("LSTM units for second LSTM", 1, 40, 40, 5)
    lstm_units_03 = st.number_input("LSTM units for third LSTM", 1, 40, 40, 5)
    lstm = [lstm_units_01, lstm_units_02, lstm_units_03]

"""
## Dense layers
"""
dense_layers = st.slider("Dense Layers", 1, 3, 1, 1)
if dense_layers == 1:
    dense_units_01 = st.number_input("Dense units for first Dense", 1, 512, 512, 1)
    dense = [dense_units_01]
elif dense_layers == 2:
    dense_units_01 = st.number_input("Dense units for first Dense", 1, 512, 512, 1)
    dense_units_02 = st.number_input("Dense units for second Dense", 1, 256, 256, 5)
    dense = [dense_units_01, dense_units_02]
elif dense_layers == 3:
    dense_units_01 = st.number_input("Dense units for first Dense", 1, 512, 512, 1)
    dense_units_02 = st.number_input("Dense units for second Dense", 1, 256, 256, 1)
    dense_units_03 = st.number_input("Dense units for third Dense", 1, 128, 128, 1)
    dense = [dense_units_01, dense_units_02, dense_units_03]


def create_model(lstm, dense):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=[None, 12]))
    for units in lstm:
        # model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units, return_sequences=True)))
        model.add(tf.keras.layers.LSTM(units, return_sequences=True))
    for units in dense:
        model.add(tf.keras.layers.Dense(units, activation='relu'))
    model.add(tf.keras.layers.Dense(6, activation='softmax'))
    model.summary()
    return model


def train_model(model, epochs):

    class MyCallBack(tf.keras.callbacks.Callback):
        epoch_placeholder = st.empty()
        my_bar = st.progress(0)
        placeholder = st.empty()
        placeholder2 = st.empty()
        placeholder3 = st.empty()
        epoch_ = 0
        
        def on_epoch_begin(self, epoch, logs):
            self.epoch_placeholder.write("""Epoch #{}""".format(epoch))

        def on_batch_end(self, batch, logs):
            self.my_bar.progress(batch/batches)
            self.placeholder.write("""
                ## Loss 
                {loss}                
                ## Accuracy
                {acc}
                """.format(loss=logs.get("loss"), acc=logs.get("accuracy"))
            )
        
        def on_test_begin(self, logs):
            self.placeholder2.write("""
            ## Running over the validation set for epoch {}, please be patient...
            """.format(self.epoch_))
        
        def on_test_end(self, logs):
            self.placeholder2.write("""
            ## Waiting for next training epoch to finish...
            """)

        def on_epoch_end(self, epoch, logs):
            self.epoch_ = epoch
            self.placeholder3.write(
                """
               ## Validation Loss 
                {loss}                
                ## Validation Accuracy
                {acc}
                """.format(loss=logs.get("val_loss"), acc=logs.get("val_accuracy"))
            )
            


    my_cb = MyCallBack()

    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(),
        loss=tf.keras.losses.categorical_crossentropy,
        metrics=[
            tf.keras.metrics.Accuracy('accuracy'), 
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )

    history = model.fit(
        train,
        epochs=epochs,
        validation_data=test,
        callbacks=[my_cb],
        # class_weight=class_weights,
        # sample_weight=class_weights,

    )

    return history


model = create_model(lstm, dense)
"""
# Debugging the created model
The following shows a layer by layer depiction of the created model. As you can see there is an extra Dense layer with 6 units which correspond to the 6 different classes we are trying to predict.
"""
model.summary(print_fn=st.text)


"""
# Finish the configuration
To finish the configuration, select the number of epochs you want to train for
"""
epochs = st.number_input("Number of epochs: ", 1, 10, 2, 1)

# """
# Important note:

# Keep in mind that if the dataset was kept as is we will have to apply some sort of class imbalance strategy.

# """
# class_weights = class_weight.compute_class_weight('balanced',
#                                                  np.unique(labels),
#                                                  labels)

"""
## Class weights

"""
# dict_class_weights = dict(enumerate(class_weights))
# dict_class_weights

# class_weights_matrix = np.zeros((batches, 6))
# class_weights_matrix[:,0] = dict_class_weights[0]
# class_weights_matrix[:,1] = dict_class_weights[1]
# class_weights_matrix[:,2] = dict_class_weights[2]
# class_weights_matrix[:,3] = dict_class_weights[3]
# class_weights_matrix[:,4] = dict_class_weights[4]
# class_weights_matrix[:,5] = dict_class_weights[5]

"""
I invite you to change the number of samples and see how the weights change, what happens if all classes have the same number of samples? 

---
"""

run_training = st.button("Start training")

# run_training = True
if run_training:
    
    history = train_model(model, epochs)

    # fig = plt.figure(figsize=(10,10))
    # plt.plot(range(epochs), history.history['loss'])
    # plt.plot(range(epochs), history.history['val_loss'])
    # plt.show()
    fig = go.Figure(data=go.Scatter(x=np.array(range(epochs)), y=history.history['loss'], name='loss'))
    fig.add_trace(go.Scatter(x=np.array(range(epochs)), y=history.history['val_loss'], name='val_loss'))
    st.plotly_chart(fig)

    # fig = plt.figure(figsize=(10,10))
    # plt.plot(range(epochs), history.history['accuracy'])
    # plt.plot(range(epochs), history.history['val_accuracy'])
    # plt.show()
    fig = go.Figure(data=go.Scatter(x=np.array(range(epochs)), y=history.history['accuracy'], name='accuracy'))
    fig.add_trace(go.Scatter(x=np.array(range(epochs)), y=history.history['val_accuracy'], name='val_accuracy'))
    st.plotly_chart(fig)


# forecast = []
# forecast.append(model.predict(test_dataset))

# plot_series(time_valid, x_valid)
# plot_series(time_valid[:442], np.array(forecast)[0,:,0])

# TODO: class weights for the loss and number of samples have to be proportional across all classes
