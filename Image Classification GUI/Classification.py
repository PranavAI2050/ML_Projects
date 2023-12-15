import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import plotly.express as px
import tensorflow as tf
from io import StringIO
import sys
plot_model = tf.keras.utils.plot_model
from PIL import Image
from keras.utils import plot_model
image = tf.keras.preprocessing.image
preprocess_input = tf.keras.applications.inception_v3.preprocess_input
rgb_to_grayscale = tf.image.rgb_to_grayscale

BatchNormalization = tf.keras.layers.BatchNormalization
Conv2D = tf.keras.layers.Conv2D
BatchNormalization = tf.keras.layers.BatchNormalization
Activation = tf.keras.layers.Activation
Flatten = tf.keras.layers.Flatten
Dropout = tf.keras.layers.Dropout
Dense = tf.keras.layers.Dense


@st.cache_data
def get_images():
    return pd.read_csv(os.path.join(os.getcwd(), 'Data/train.csv'))


# configuration of the page
st.set_page_config(layout="wide")
# load dataframes
train = get_images()


def Dataset(data):
    Y_train = data["label"]
    X_train = data.drop(labels=["label"], axis=1)
    X_train = X_train / 255.0
    X_train = X_train.values.reshape(-1, 28, 28, 1)
    return X_train, Y_train


X_train, Y_train = Dataset(train)

tab0, tab1, tab2, tab3, tab4 = st.tabs(["ABOUT", "DATASET", "MODEL", 'TRAIN', 'PREDICT'])

with tab0:
    st.title("Introduction to Image Classification")
    st.markdown(
        "<h4 style = 'test-align:center;'>Welcome to the world of image classification! Image classification is a fascinating field    in computer vision where machines learn to categorize and identify objects in images as shown below.<h4>",
        unsafe_allow_html=True
    )
    st.image('Images/Classif_cat_dog.png', caption="Your Image", use_column_width=True)
    st.markdown(
        "<h4 style = 'test-align:center;'>If you're interested in image classification but lack coding experience, we've got you covered. Simply follow our user-friendly interface instructions to effortlessly build your first machine learning model.<h4>",
        unsafe_allow_html=True
    )

with tab1:
    st.title('Introduction to Image Classification')
    st.markdown("<h1 style='text-align: left; color: blue;'>Create Your Own  Dataset</h1>", unsafe_allow_html=True)
    cols = st.columns(2)
    with cols[0]:
        st.image('Images/Mnist_example.png', caption="Your Image", use_column_width=True)
    with cols[1]:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            "<h3 style='text-align:left;'>For your initial model, we're going to create a digit recognition model. Let's begin by gaining insights into the data through visualizations. In this process, we are going to use the MNIST Dataset</h3>",
            unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(
            """<h5 style='text-align:left;'>The MNIST dataset is a collection of 28x28 grayscale images of handwritten digits (0-9), commonly used for training and evaluating machine learning models for digit recognition.</h5>""",
            unsafe_allow_html=True)


    def plot_images(images, labels, num):
        num_displayed = int(num - num % 2)
        num_rows, num_cols = 2, num_displayed // 2
        random_indices = np.random.choice(len(images), num_displayed, replace=False)
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 4.5))
        fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.3, hspace=0.3)
        for i, index in enumerate(random_indices):
            ax = axes[i // num_cols, i % num_cols]
            ax.set_title(f"Label: {labels[index]}", fontsize=10)
            ax.imshow(X_train[index], cmap=plt.cm.binary)
            ax.axis('off')
        st.pyplot(fig)


    col_plots = st.columns(2)

    with col_plots[0]:
        st.markdown("<h1>MNIST Images with Labels of Digits</h1>", unsafe_allow_html=True)
        st.markdown("""<h4>Select Number of samples to display </h4>""", unsafe_allow_html=True)
        num = st.slider("", min_value=1, max_value=30, value=10)

        if st.checkbox("Display Images"):
            plot_images(X_train, Y_train, num)

    def plot_distribution(X_train, Y_train, num_samples):
        color_map = {
            0: 'red',
            1: 'green',
            2: 'blue',
            3: 'purple',
            4: 'orange',
            5: 'cyan',
            6: 'magenta',
            7: 'yellow',
            8: 'lime',
            9: 'brown'
        }
        random_indices = np.random.choice(len(X_train), num_samples, replace=False)
        X_train_sub = X_train[random_indices]
        y_train_sub = Y_train[random_indices]
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_train_sub.reshape((-1, 784)))
        pca_df = pd.DataFrame({
            'PC1': X_pca[:, 0],
            'PC2': X_pca[:, 1],
            'Label': y_train_sub
        })
        pca_df['color'] = pca_df.Label.map(color_map)
        fig, axs = plt.subplots(figsize = (10,6))
        for i in range(10):
            v = (pca_df.Label == i)
            pca_sub = pca_df[v]
            plt.scatter(pca_sub.PC1.values,pca_sub.PC2.values, color= pca_sub.color.values[1], label =pca_sub.Label.values[1] )
        plt.legend()
        plt.xlabel("Pca Component 1")
        plt.ylabel("Pca Component 2")
        plt.title("Plot Showing the Distribution of Digits Represented by Diff Colors")
        plt.tight_layout()
        return fig


    with col_plots[1]:
        st.title("Interactive PCA Plot with Labels")

        st.markdown("""<h4>Select Number of samples to display </h4>""", unsafe_allow_html=True)
        num_samples = st.slider("  ", min_value=100, max_value=len(X_train),
                                value=200)

        # Call the plot_distribution function and display the returned figure
        if st.checkbox("Visualize The Data"):
            fig = plot_distribution(X_train, Y_train, num_samples)
            st.pyplot(fig)

    st.markdown("<h1 style='text-align: left; color: blue;'>Test Train Split</h1>", unsafe_allow_html=True)
    st.markdown(
        "<h3 style='text-align: left;'>Train-test split is a technique to partition a dataset into two subsets, one for training a machine learning model and the other for Validating the model's performance, typically used to evaluate the model's generalization</h3>",
        unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: left;'> In this case, we are splitting the data in a stratified manner to maintain the proportion of labels in the validation and training sets</h3>",
    unsafe_allow_html = True)


    def split_data(X_train, Y_train, split_ratio, Datasamples):
        random_indices = np.random.choice(len(X_train), Datasamples, replace=False)
        images = X_train[random_indices]
        labels = Y_train[random_indices]
        images_train, images_test, labels_train, labels_test = train_test_split(images, labels,
                                                                                test_size=1 - split_ratio,
                                                                                random_state=42, shuffle=True,
                                                                                stratify=labels)
        return images_train, images_test, labels_train, labels_test, len(images_train), len(images_test)


    cols_splits = st.columns(2)
    with cols_splits[0]:
        st.markdown("<h4>Data Samples For Training</h4>", unsafe_allow_html=True)
        num_s = st.slider("Select the number of samples for training", min_value=100, max_value=len(X_train),
                          value=200)

    with cols_splits[1]:
        st.markdown("<h4>Split Ratio</h4>", unsafe_allow_html=True)
        num_ratio = st.slider("Select the split ratio", min_value=0.1, max_value=0.9, value=0.5, step=0.01,
                              format="%f")
    summary_cols = st.columns(3)
    if st.checkbox("Generate Data For Training And Testing"):
        Train_X, Test_X, Train_Y, Test_Y, num_train_samples, num_test_samples = split_data(X_train, Y_train,
                                                                                           num_ratio, num_s)
        with summary_cols[1]:
            st.title("Train and Test Data Summary")
            data = pd.DataFrame({'Train_Data': [num_train_samples], "Test_Data": [num_test_samples]})
            st.table(data)

with tab2:
    st.title('Build Your own Model')
    st.markdown("<h2 style='text-align: left; color: blue;'>Design The Neural Network</h2>", unsafe_allow_html=True)
    st.markdown(
        "<h4>Create an Artificial Neural Network (ANN) model effortlessly by selecting layers and configuring other requirements , An Artificial Neural Network (ANN) is a machine learning model inspired by the human brain's neural structure, used for tasks like classification and regression</h4>",
        unsafe_allow_html=True)
    st.markdown(
        "<h2 style='text-align: left; color: blue;'>Below Image shows a Neural Network for Image Classification</h2>",
        unsafe_allow_html=True)


    def model(layer_configs):
        model = tf.keras.Sequential()
        model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))
        for config in layer_configs:
            layer_type = config['layer_type']

            if layer_type == 'Conv2D':
                filters = config['filters']
                kernel_size = config['kernel_size']
                activation_function = config['activation_function']

                model.add(Conv2D(filters, kernel_size=kernel_size, activation=activation_function))

            elif layer_type == 'Dense':
                dense_units = config['dense_units']
                activation_function = config['activation_function']

                model.add(Dense(dense_units, activation=activation_function))

            elif layer_type == 'Dropout':
                dropout_rate = config['dropout_rate']

                model.add(Dropout(dropout_rate))

            elif layer_type == 'BatchNormalization':
                model.add(BatchNormalization())

            elif layer_type == 'Flatten':
                model.add(Flatten())
        model.add(Dense(10, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model


    def build_ann_model(num_layers):
        build = False
        # User input for number of layers and units
        layers = st.columns(num_layers)
        layer_configs = []
        for i in range(num_layers):
            with layers[i]:

                layer_type = st.selectbox(f"TYPE OF ANN FOR LAYER {i + 1} ",
                                          ['Conv2D', 'Dropout', 'BatchNormalization', 'Dense', 'Flatten'])
                if layer_type == 'Conv2D':
                    filters = st.number_input(f"filters in Layer {i + 1}", min_value=32, max_value=128, value=32)
                    kernel_size = st.number_input(f"kernel_size for Layer{i + 1}", min_value=1, max_value=4,
                                                  value=3)
                    activation_functions = st.selectbox(f"Activation Function Layer {i + 1} ",
                                                        ["relu", "sigmoid", "tanh"])
                    layer_config = {'layer_type': layer_type, 'filters': filters, 'kernel_size': kernel_size,
                                    'activation_function': activation_functions, 'dense_units': None,
                                    'dropout_rate': None}

                elif layer_type == 'Dense':
                    dense_units = st.number_input(f"filters in Layer {i + 1}", min_value=128, max_value=360,
                                                  value=32)
                    activation_functions = st.selectbox(f"Activation Function Layer {i + 1} ",
                                                        ["relu", "sigmoid", "tanh", 'linear'])
                    layer_config = {'layer_type': layer_type, 'filters': None, 'kernel_size': None,
                                    'activation_function': activation_functions, 'dense_units': dense_units,
                                    'dropout_rate': None}
                elif layer_type == 'Dropout':
                    dropout_rate = st.number_input(f"filters in Layer {i + 1}", min_value=0.1, max_value=0.4,
                                                   value=0.4)
                    layer_config = {'layer_type': layer_type, 'filters': None, 'kernel_size': None,
                                    'activation_function': None, 'dense_units': None,
                                    'dropout_rate': dropout_rate}
                else:
                    layer_config = {'layer_type': layer_type, 'filters': None, 'kernel_size': None,
                                    'activation_function': None, 'dense_units': None,
                                    'dropout_rate': None}
                layer_configs.append(layer_config)
        return layer_configs


    st.image('Images/CNN.png', caption="Your Image", use_column_width=True)
    st.markdown(
        "<h4  style = 'text-align: left;'>Build your machine learning model effortlessly by selecting model specifications. The initial layer is prebuilt as a Conv2D layer with input shape (28, 28, 1) and output shape 10 with a sigmoid activation for Your convenience</h4>",
        unsafe_allow_html=True)

    cols_layer = st.columns(3)
    st.markdown(
        "<h5  style = 'text-align: left;'>Make sure the model should Have a  Flatten Layer at the End and not in Between for Compatibility</h5>",
        unsafe_allow_html=True)
    st.markdown(
        "<h5  style = 'text-align: left;'>You Can Tweak with different layers in between and explores what gives Better Results</h5>",
        unsafe_allow_html=True)

    with cols_layer[1]:
        num_layers = st.slider("Number of Layers", min_value=2, max_value=10, value=1)
    layer_configs = build_ann_model(num_layers)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.checkbox('Create Model'):
        ann_model = model(layer_configs)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.checkbox('show model Structure'):
        # Save the original sys.stdout for later restoration
        original_stdout = sys.stdout

        # Redirect sys.stdout to capture the summary text
        sys.stdout = StringIO()

        # Print the summary to the captured sys.stdout
        ann_model.summary()

        # Get the captured text
        summary_text = sys.stdout.getvalue()

        # Restore sys.stdout to its original state
        sys.stdout = original_stdout

        # Display the summary in the Streamlit app
        st.text(summary_text)


with tab3:
    st.title("Train Your Neural Network")
    st.markdown(
        "<h4>Model training is the process of teaching a machine learning model to recognize patterns or relationships in data.</h4>",
        unsafe_allow_html=True)
    st.markdown("<h5>So let's train your own neural network for digits classification</h5>", unsafe_allow_html=True)
    st.markdown("<h5>You can try different Tranning Parameters by selecting from below</h5>", unsafe_allow_html=True)
    param_cols = st.columns(2)
    with param_cols[0]:
        st.markdown("<h4 style = 'text-align :center'>Setect Tranning Parameters</h4>", unsafe_allow_html=True)
        epochs = st.number_input("Epochs", min_value=1, max_value=10, value=3)
        batch_size = st.number_input("Batch_size", min_value=16, max_value=48, value=32)
        learning_rate = st.number_input("Learning rate", min_value=0.001, max_value=0.1, value=0.05)
        optimizer = st.selectbox('Optimizer', ['adam', 'sgd', 'rmsprop'])

    curr_model = model(layer_configs)
    if st.checkbox("compile Model"):
        curr_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


    if st.checkbox('Train Model'):
        Train_Y = to_categorical(Train_Y, num_classes=10)
        Test_Y = to_categorical(Test_Y, num_classes=10)
        history = curr_model.fit(Train_X, Train_Y, epochs=epochs, batch_size=batch_size,
                                     validation_data=(Test_X, Test_Y))
        st.success('Model trained successfully!')
    model_tranning_plots = st.columns(4)


    def plot_model_performance(history):

        final_training_loss = history.history['loss'][-1]
        final_validation_loss = history.history['val_loss'][-1]
        final_training_accuracy = history.history['accuracy'][-1]
        final_validation_accuracy = history.history['val_accuracy'][-1]

        with model_tranning_plots[0]:
            st.subheader('Training Loss')
            st.text(f"Final Training Loss (Last Epoch): {final_training_loss:.4f}")
            fig, ax = plt.subplots()
            ax.plot(history.history['loss'])
            st.pyplot(fig)

        with model_tranning_plots[1]:
            st.subheader('Validation Loss')
            st.text(f"Final Validation Loss (Last Epoch): {final_validation_loss:.4f}")
            fig, ax = plt.subplots()
            ax.plot(history.history['val_loss'])
            st.pyplot(fig)

        with model_tranning_plots[2]:
            st.subheader('Training Accuracy')
            st.text(f"Final Training Accuracy (Last Epoch): {final_training_accuracy:.4f}")
            fig, ax = plt.subplots()
            ax.plot(history.history['accuracy'])
            st.pyplot(fig)

        with model_tranning_plots[3]:
            st.subheader('Validation Accuracy')
            st.text(f"Final Validation Accuracy (Last Epoch): {final_validation_accuracy:.4f}")
            fig, ax = plt.subplots()
            ax.plot(history.history['val_accuracy'])
            st.pyplot(fig)
        st.subheader('Training Loss (Epoch-wise)')
        epoch_numbers = np.arange(1, len(history.history['loss']) + 1)
        st.bar_chart(list(zip(epoch_numbers, history.history['loss'])))


    if st.checkbox("Show Model Performance"):
        plot_model_performance(history)

with tab4:
    st.title("Make Predictions on your Data")
    st.markdown("<h3 style = 'text-align : left; color:blue;'></h3>", unsafe_allow_html=True)
    pred_cols = st.columns(2)
    with pred_cols[0]:
        uploaded_file = st.file_uploader("Upload Image")

    with pred_cols[1]:
        if st.checkbox("Make Prediction"):
            def preprocess_image(img_path):
                img = image.load_img(img_path, target_size=(28, 28))
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = rgb_to_grayscale(img_array)
                img_array = preprocess_input(img_array)
                return img_array


            def predict(image_path):

                img_array = preprocess_image(image_path)
                predictions = curr_model.predict(img_array)[0]
                predicted_class_index = np.argmax(predictions)
                return predicted_class_index + 1, predictions[predicted_class_index]


            if uploaded_file is not None:
                st.image(uploaded_file, caption='Uploaded Image.', use_column_width=False)
                st.write("")
                st.write("classifying.....")
                predictions, confidence_value = predict(uploaded_file)
                st.write(f"Predictions: Label is   {predictions}     with Probability of ({confidence_value:.2f})")


