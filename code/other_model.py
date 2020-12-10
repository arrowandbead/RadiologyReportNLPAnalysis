from transformers import AutoTokenizer, TFAutoModel
from sklearn.metrics import classification_report
import tensorflow as tf
import numpy as np
import random
from preprocessing import get_data

SEQ_LEN = 128
SPLIT = 0.8

class MSNR(tf.keras.Model):
    def __init__(self, impressions, labels, biobert):
        super(MSNR, self).__init__()
        self.impressions = impressions
        self.labels = labels
        self.biobert = biobert.biobert_model

        # Hyperparameters
        self.optimizer = tf.keras.optimizers.Adam(0.01)
        self.loss = tf.keras.losses.CategoricalCrossentropy()
        self.batch_size = 32
        self.epochs = 20
        
    def call(self, input_ids, input_masks):
        # print("input ids:", input_ids)
        # biobert_embeddings = []
        # for i in range(len(input_ids)):
        #     print(len(input_ids[i]))
        #     print(biobert_embeddings(input_masks[i]))
        embeddings = self.biobert(input_ids, attention_mask=input_masks)[0]
        X = tf.keras.layers.GlobalMaxPool1D()(embeddings)  # reduce tensor dimensionality
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Dense(128, activation='relu')(X)
        X = tf.keras.layers.Dropout(0.1)(X)
        probabilities = tf.keras.layers.Dense(7, activation='softmax', name='outputs')(X)
        
        return probabilities

    def loss_function(self, probabilities, labels):
        """
            Input: predictions 
            Output: categorical cross-entropy loss
        """

        # can also use keras's categorical cross entropy

        log_vals = [np.log(probabilities) if probability != 0 else 0 for probability in probabilities]
        cross_entropy = -np.sum(labels * log_vals) / len(probabilities)
        return cross_entropy

    def accuracy_function(self, predictions, labels):
        pass

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
        shuffled_indices = np.arange(len(self.impressions))
        np.random.shuffle(shuffled_indices)
        ids = ids[shuffled_indices]
        masks = masks[shuffled_indices]
        encoded_labels = encoded_labels[shuffled_indices]

        train_size = int(len(self.impressions) * 0.8)

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
    print("inside train")
    # train in batches
    for i in range(0, len(train_ids), model.batch_size):
        batch_ids = train_ids[i:i + model.batch_size]
        batch_masks = train_masks[i:i + model.batch_size]
        batch_y = train_labels[i:i + model.batch_size]
        with tf.GradientTape() as tape:
            probabilities = model.call(batch_ids, batch_masks)
            curr_loss = model.loss_function(probabilities, batch_y)
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
    
    for i in range(0, len(test_ids), model.batch_size):
        batch_ids = test_ids[i:i + model.batch_size]
        batch_masks = test_masks[i:i + model.batch_size]
        batch_y = test_labels[i:i + model.batch_size]
        with tf.GradientTape() as tape:
            probabilities = model.call(batch_ids, batch_masks)
            curr_loss = model.loss_function(probabilities, batch_y)
        gradients = tape.gradient(curr_loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # calculate accuracy instead of loss
    accuracy = 0
    
    # calculate perplexity
    for i in range(0, len(test_ids), model.batch_size):
        batch_ids = test_ids[i:i + model.batch_size]
        batch_masks = test_masks[i:i + model.batch_size]
        batch_y = test_labels[i:i + model.batch_size]
        probs = model.call(batch_ids, batch_masks)
        weight = float(len(test_ids)) / len(test_ids)
        # avg_loss += weight * model.loss(probs, batch_y)
    # return np.exp(avg_loss)
    return accuracy


def main():

    file_name = "giacomomiolo/biobert_reupload"
    impressions, labels = get_data()
    biobert = BioBERT(file_name, impressions, labels)
    train_data, test_data = biobert.tokenize_and_split_data()
    model = MSNR(impressions, labels, biobert)
    train(model, train_data[0], train_data[1], train_data[2])
    print(test(model, test_data[0], test_data[1], test_data[2]))

if __name__ == '__main__':
    main()
