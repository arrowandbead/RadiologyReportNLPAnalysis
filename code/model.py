from transformers import AutoTokenizer, TFAutoModel
from sklearn.metrics import classification_report
import tensorflow as tf
import numpy as np
import random
from preprocessing import get_data
import matplotlib.pyplot as plt

SEQ_LEN = 128
SPLIT = 0.8

class MSNR(tf.keras.Model):
    def __init__(self, impressions, labels, biobert):
        super(MSNR, self).__init__()
        self.impressions = impressions
        self.labels = labels
        self.biobert = biobert.biobert_model

        # Hyperparameters
        self.optimizer = tf.keras.optimizers.Adam(0.001) 
        self.loss = tf.keras.losses.CategoricalCrossentropy()
        self.batch_size = 32
        self.epochs = 20
        self.cce = tf.keras.losses.CategoricalCrossentropy()

        # Layers
        self.global_average_pool = tf.keras.layers.GlobalAveragePooling1D()
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.dense = tf.keras.layers.Dense(128, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.1)
        self.dense_sftmx = tf.keras.layers.Dense(7, activation='softmax', name='outputs')
        self.cat_acc = tf.keras.metrics.CategoricalAccuracy()
        
    def call(self, input_ids, input_masks):
        """
            Input: input ids and input masks for a batch of examples
            Output: probabilities for each class for each example in this batch
        """
        embeddings = self.biobert(input_ids, attention_mask=input_masks)[0]
        X = self.global_average_pool(embeddings)  
        X = self.batch_norm(X)
        X = self.dense(X)
        X = self.dropout(X)
        probabilities = self.dense_sftmx(X)
        return probabilities

    def loss_function(self, probabilities, labels):
        """
            Input: predictions 
            Output: categorical cross-entropy loss
        """
        return self.cce(labels, probabilities)

    def accuracy_function(self, predictions, labels):
        """
            Input: predictions and true labels for a batch of examples
            Output: number of correctly predicted examples in this batch, 
                    (7,) array of correct examples for each class, (7,) array
                    of number of examples for each class
            Purpose: Our hand-written accuracy function to sanity-check Categorical Accuracy
                    and record per-class accuracy
        """
        decoded_predictions = np.argmax(predictions, axis=1) # decode predictions
        decoded_labels = np.argmax(labels, axis=1) # decode labels

        correct_overall = 0
        correct_per_class = np.zeros(7)
        examples_per_class = np.zeros(7)
        for i in range(len(decoded_predictions)):
            if decoded_predictions[i] == decoded_labels[i]:
                correct_overall += 1
                correct_per_class[decoded_predictions[i]] += 1
            examples_per_class[decoded_labels[i]] += 1    
        return correct_overall, correct_per_class, examples_per_class

class BioBERT():
    def __init__(self, file_name, impressions, labels):
        self.biobert_tokenizer = AutoTokenizer.from_pretrained(file_name)
        self.biobert_model = TFAutoModel.from_pretrained(file_name)
            
        self.impressions = impressions
        self.labels = labels
        
    def tokenize(self, impression):
        """
            Input: sentence as string
            Output: tokenized sentence array (including special tokens) and attention mask as array (1s for word and 0 o.w.)
        """
        tokens = self.biobert_tokenizer.encode_plus(impression, max_length=SEQ_LEN,
                                   truncation=True, padding='max_length',
                                   add_special_tokens=True, return_attention_mask=True,
                                   return_token_type_ids=False, return_tensors='tf')

        return tokens['input_ids'], tokens['attention_mask']

    def tokenize_and_split_data(self):
        """
            Input: None
            Output: biobert embeddings of impressions
        """
        # one hot encode labels
        encoded_labels = np.zeros((len(self.impressions), 7))
        encoded_labels[np.arange(len(self.impressions)), np.array(self.labels) - 1] = 1

        # initialize two arrays for input tensors and tokenize impressions
        ids = np.zeros((len(self.impressions), SEQ_LEN))
        masks = np.zeros((len(self.impressions), SEQ_LEN))
        for i, impression in enumerate(self.impressions):
            ids[i, :], masks[i, :] = self.tokenize(impression)

        # shuffle
        np.random.seed(0)
        shuffled_indices = np.arange(len(self.impressions))
        np.random.shuffle(shuffled_indices)
        ids = ids[shuffled_indices]

        # convert to tensors
        ids = tf.convert_to_tensor(ids,  dtype='int32')
        masks = masks[shuffled_indices]
        masks = tf.convert_to_tensor(masks, dtype='int32')
        encoded_labels = encoded_labels[shuffled_indices]
        encoded_labels = tf.convert_to_tensor(encoded_labels)

        # split into train-test
        train_size = int(len(self.impressions) * SPLIT)
        train_ids = ids[:train_size]
        train_mask = masks[:train_size]
        train_labels = encoded_labels[:train_size]
        train_data = [train_ids, train_mask, train_labels]

        test_ids = ids[train_size:]
        test_mask = masks[train_size:]
        test_labels = encoded_labels[train_size:]
        test_data = [test_ids, test_mask, test_labels]

        return train_data, test_data


def train(model, train_ids, train_masks, train_labels):
    """
    Runs through one epoch - all training examples.

    :param model: the initilized model to use for forward and backward pass
    :param train_inputs: train inputs (all inputs for training) of shape (num_inputs,)
    :param train_labels: train labels (all labels for training) of shape (num_labels,)
    :return: None
    """
    # train in batches
    for i in range(0, len(train_ids), model.batch_size):
        batch_ids = train_ids[i:i + model.batch_size]
        batch_masks = train_masks[i:i + model.batch_size]
        batch_y = train_labels[i:i + model.batch_size]
        with tf.GradientTape() as tape:
            probabilities = model.call(batch_ids, batch_masks)
            curr_loss = model.loss_function(probabilities, batch_y)
            model.cat_acc.update_state(batch_y, probabilities)
        gradients = tape.gradient(curr_loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def test(model, test_ids, test_masks, test_labels):
    """
    Runs through one epoch - all testing examples

    :param model: the trained model to use for prediction
    :param test_inputs: train inputs (all inputs for testing) of shape (num_inputs,)
    :param test_labels: train labels (all labels for testing) of shape (num_labels,)
    :returns: perplexity of the test set
    """
    # to calculate test accuracy with
    correct_per_class = np.zeros(7)
    examples_per_class = np.zeros(7)
    accuracy = 0
    
    # test in batches
    for i in range(0, len(test_ids), model.batch_size):
        batch_ids = test_ids[i:i + model.batch_size]
        batch_masks = test_masks[i:i + model.batch_size]
        batch_y = test_labels[i:i + model.batch_size]
        probs = model.call(batch_ids, batch_masks)
        weight = float(len(batch_ids)) / len(test_ids)
        batch_correct_overall, batch_correct_per_class, batch_examples_per_class = model.accuracy_function(probs, batch_y)
        model.cat_acc.update_state(batch_y, probs) # Keras Categorical Accuracy
        accuracy += weight * batch_correct_overall / len(batch_y)
        correct_per_class += batch_correct_per_class
        examples_per_class += batch_examples_per_class
    accuracy_per_class = [correct_per_class[i] / examples_per_class[i] if examples_per_class[i] != 0 else 0 for i in range(7)]
    return accuracy, accuracy_per_class, examples_per_class

def main():

    # load BioBERT from Hugging Face
    file_name = "giacomomiolo/biobert_reupload"
    impressions, labels = get_data()
    biobert = BioBERT(file_name, impressions, labels)

    # get train and test data
    train_data, test_data = biobert.tokenize_and_split_data()
    model = MSNR(impressions, labels, biobert)
    model.layers[0].trainable = False # freeze BioBERT layer to only train our classifier
    epoch_accuracy = []
    per_class_epoch_accuracy = []

    for i in range(model.epochs):
        train(model, train_data[0], train_data[1], train_data[2])
        print("epoch:", i, "/ 19")

    # print accuracies
    train_accuracy = model.cat_acc.result().numpy()
    print("Keras Categorical Accuracy (train)", train_accuracy)
    results = test(model, test_data[0], test_data[1], test_data[2])
    print("per class accuracy:", results[1])
    print("# of examples per class:", results[2])
    test_accuracy = model.cat_acc.result().numpy()
    print("Keras Categorical Accuracy (test)", test_accuracy)
    
if __name__ == '__main__':
    main()
