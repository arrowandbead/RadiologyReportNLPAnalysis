from transformers import AutoTokenizer, TFAutoModel
import tensorflow as tf
import numpy as np
from preprocessing import get_data

SEQ_LEN = 128

class MSNR():
    def __init__(self):
        # super(MSNR, self).__init__()

        # Initializa model and tokenizer
        file_name = "giacomomiolo/biobert_reupload"
        self.biobert_tokenizer = AutoTokenizer.from_pretrained(file_name)
        self.biobert = TFAutoModel.from_pretrained(file_name)

        # class variables
        self.num_labels = 7
        self.impressions, self.labels = get_data()

        # Input and layers
        # self.dense_layer = tf.keras.layers.Dense(self.num_labels, activation='softmax', name='outputs')  # adjust based on number of sentiment classes
        self.input_ids = tf.keras.layers.Input(shape=(SEQ_LEN,), name='input_ids', dtype='int32')
        self.mask = tf.keras.layers.Input(shape=(SEQ_LEN,), name='attention_mask', dtype='int32')

        # Model variables
        self.optimizer = tf.keras.optimizers.Adam(0.01)
        self.loss = tf.keras.losses.CategoricalCrossentropy()
        self.accuracy = tf.keras.metrics.CategoricalAccuracy('accuracy')
    
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

    def get_biobert_embeddings(self):
        """
            Input: None
            Output: biobert embeddings of impressions
        """
        # one hot encode labels
        encoded_labels = np.zeros((len(self.impressions), self.num_labels))
        encoded_labels[np.arange(len(self.impressions)), np.array(self.labels) - 1] = 1

        # initialize two arrays for input tensors and tokenize impressions
        Xids = np.zeros((len(self.impressions), SEQ_LEN))
        Xmask = np.zeros((len(self.impressions), SEQ_LEN))
        for i, impression in enumerate(self.impressions):
            Xids[i, :], Xmask[i, :] = self.tokenize(impression)

        # create TF dataset object
        dataset = tf.data.Dataset.from_tensor_slices((Xids, Xmask, encoded_labels))
        dataset = dataset.map(self.map_func)  # apply the mapping function

        # shuffle and batch dataset
        dataset = dataset.shuffle(454).batch(32)

        # create train and test sets
        train_data = dataset.take(round(len(self.impressions) * 0.8))
        test_data = dataset.skip(round(len(self.impressions) * 0.8))
        print(len(train_data))
        print(len(test_data))
        # # free space
        # del dataset

        return self.biobert(self.input_ids, attention_mask=self.mask)[0], train_data, test_data

    def map_func(self, input_ids, masks, labels):
        """
            Purpose: structure data for input to BERT
            Input: input_id, mask and encoded label arrays from our impressions and annotations
            Output: tuple of (dictionary mapping name to array, labels)
        """
        return {'input_ids': input_ids, 'attention_mask': masks}, labels
  

    def train_model(self):   
        """
            Purpose: Instantiates and trains the model
            Input: None
            Output: None
        """
        embeddings, train_data, test_data = self.get_biobert_embeddings()
        # outputs = self.dense_layer(embeddings)
        X = tf.keras.layers.GlobalMaxPool1D()(embeddings)  # reduce tensor dimensionality
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Dense(128, activation='relu')(X)
        X = tf.keras.layers.Dropout(0.1)(X)
        y = tf.keras.layers.Dense(7, activation='softmax', name='outputs')(X) 
        model = tf.keras.Model(inputs=[self.input_ids, self.mask], outputs=y)
        
        # freeze the BERT layer
        model.layers[2].trainable = False

        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=[self.accuracy])

        # and train it
        history = model.fit(train_data, epochs=20)
        print(history)
        results = model.evaluate(test_data)
        print(results)
    
    def get_embeddings(self, sentences):
        """
            Input: Sentences in an array (e.g. sentences = ["Sentence one.",  "Sentence two."])
            Output: sentence embeddings as array of size [1 x 768] (?)
        """
        batch = self.biobert_tokenizer(sentences, padding=True, truncation=True, return_tensors='tf')
        outputs = self.biobert(batch, output_hidden_states=True)
        hidden_states = outputs[2]
        embeddings = hidden_states[0]
        return embeddings



def main():

    m = MSNR()
    print(m.train_model())

if __name__ == '__main__':
    main()
