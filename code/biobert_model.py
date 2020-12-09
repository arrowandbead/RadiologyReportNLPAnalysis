from transformers import AutoTokenizer, TFAutoModel
from sklearn.metrics import classification_report
import tensorflow as tf
import numpy as np
import random
from preprocessing import get_data

SEQ_LEN = 128

class MSNR():
    def __init__(self):
        # super(MSNR, self).__init__()

        # Initializa model and tokenizer
        file_name = "giacomomiolo/biobert_reupload"
        bert_name = "bert-base-cased"
        self.biobert_tokenizer = AutoTokenizer.from_pretrained(file_name)
        self.biobert = TFAutoModel.from_pretrained(file_name)

        # class variables
        self.num_labels = 7
        self.impressions, self.labels = get_data()

        # Input layers
        self.input_ids = tf.keras.layers.Input(shape=(SEQ_LEN,), name='input_ids', dtype='int32')
        self.mask = tf.keras.layers.Input(shape=(SEQ_LEN,), name='attention_mask', dtype='int32')

        # Model variables
        self.optimizer = tf.keras.optimizers.Adam(0.01)
        self.loss = tf.keras.losses.CategoricalCrossentropy()
        self.accuracy = tf.keras.metrics.CategoricalAccuracy('accuracy')

        newImpressions = []
        newLabels = []

        for i, thing in enumerate(self.labels):
            if thing != 7:
                newImpressions.append(self.impressions[i])
                newLabels.append(self.labels[i])

        self.impressions = newImpressions
        self.labels = newLabels

        tog = list(zip(self.impressions, self.labels))
        random.shuffle(tog)
        self.impressions, self.labels = zip(*tog)

        self.endImp = self.impressions[-160:]
        self.impressions = self.impressions[:-160]

        self.endLab = self.labels[-160:]
        self.labels = self.labels[:-160]
        
    class RecallCallback(tf.keras.callbacks.Callback):
        def __init__(self, x, y):
            self.x = x
            self.y_true = np.array([])

            # decode one-hot labels in each batch
            for batch_of_labels in y:
                batch_labels_as_array = tf.make_ndarray(tf.make_tensor_proto(batch_of_labels))
                self.y_true = np.append(self.y_true, np.argmax(batch_labels_as_array, axis=1))
            self.reports = []

        def on_epoch_end(self, epoch, logs={}):
            y_predicted = np.argmax(np.asarray(self.model.predict(self.x)), axis=1)
            report = classification_report(self.y_true, y_predicted, labels=[0, 1, 2, 3, 4, 5, 6], output_dict=True)
            self.reports.append(report)

            print([report[str(label)]['recall'] for label in range(7)])
            # print()
            return
    
        # Utility method
        def get(self, metrics, of_class):
            return [report[str(of_class)][metrics] for report in self.reports]
    
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
        dataset = dataset.map(self.prep_for_biobert) 

        # shuffle and batch dataset
        dataset = dataset.shuffle(908).batch(32)

        # create train and test sets
        train_data = dataset.take(round(len(dataset) * 0.8))
        test_data = dataset.skip(round(len(dataset) * 0.2))

        # # free space
        # del dataset

        return self.biobert(self.input_ids, attention_mask=self.mask)[0], train_data, test_data

    def prep_for_biobert(self, input_ids, masks, labels):
        """
            Purpose: structure data for input to BioBERT
            Input: input_id, mask and encoded label arrays from our impressions and annotations
            Output: tuple of (dictionary mapping id to mask, labels)
        """
        return {'input_ids': input_ids, 'attention_mask': masks}, labels

    def get_input_ids_and_mask(self, dictionary, labels):
        """
            Purpose: grab the data without the labels 
            Input: {input_id:mask} dictionary and encoded label array
            Output: dictionary mapping id to mask
        """
        return dictionary

    def get_labels(self, dictionary, labels):
        """
            Purpose: grab the data without the labels 
            Input: {input_id:mask} dictionary and encoded label array
            Output: dictionary mapping id to mask
        """
        print("inside get labels")
        # decoded_labels = []
        # for label_set in labels:
        #     print("inside gl:", label_set)
            # print("argmax", tf.argmax(label_set, axis=1))
            # p = label_set
            # print("p as it is:", p)
            # print("p as numpy:", list(p))
            # print("p as iterator:", list(p.as_numpy_iterator()))
            # print("into ndarray", tf.make_ndarray(p))

        return labels

  
    def predict_impression(self, impression):
        ids, mask = self.tokenize(impression)
        return self.model.predict([ids, mask])

    def train_model(self):   
        """
            Purpose: Instantiates and trains the model
            Input: None
            Output: None
        """
        embeddings, train_data, test_data = self.get_biobert_embeddings()
        X = tf.keras.layers.GlobalMaxPool1D()(embeddings)  # reduce tensor dimensionality
        # print(X.shape)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Dense(128, activation='relu')(X)
        X = tf.keras.layers.Dropout(0.1)(X)
        y = tf.keras.layers.Dense(7, activation='softmax', name='outputs')(X) 
        model = tf.keras.Model(inputs=[self.input_ids, self.mask], outputs=y)


        # X = tf.keras.layers.GlobalAveragePooling1D()(embeddings, mask)  # reduce tensor dimensionality
        # X = tf.keras.layers.BatchNormalization()(X)
        # X = tf.keras.layers.Dense(768)(X)
        # X=  tf.keras.layers.ThresholdedReLU(theta=0.1)(X)
        # X = tf.keras.layers.Dropout(0.75)(X)
        # y = tf.keras.layers.Dense(6, activation='softmax', name='outputs')(X)
        
        # freeze the BERT layer
        model.layers[2].trainable = False


        class_report = MSNR.RecallCallback(train_data.map(self.get_input_ids_and_mask), train_data.map(self.get_labels))

        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=[self.accuracy])

        # and train it
        history = model.fit(train_data, epochs=20, callbacks=[class_report]) 
        print(history)
        p = model.predict(train_data.map(self.get_input_ids_and_mask))
        print(p[0])
        results = model.evaluate(test_data)
        print("results", results)
        print("class report:")
        print(classification_report(test_data))

        # correct= 0
        # total = len(self.endImp)
        # labelMap = {}
        # for i in range(1,7):
        #     labelMap[i] = [0.0,0.0]

        # for i in range(len(self.endImp)):


        #     prediction = self.predict_impression(self.endImp[i])

        #     if tf.math.argmax(prediction[0]).numpy() + 1 == self.endLab[i]:
        #         correct += 1
        #         labelMap[self.endLab[i]][0] += 1
        #     labelMap[self.endLab[i]][1] += 1
        # print('\n')
        # print(correct/total)
        # print('\n')
        # for thing in labelMap:
        #     if labelMap[thing][1] != 0:
        #         print(str(thing) + " : " + str(labelMap[thing][0]/labelMap[thing][1]))
        #     else:
        #         print(str(thing) + " : " + "N/A")
        #     print(labelMap[thing][1])
        #     print(" ")


def main():

    m = MSNR()
    print(m.train_model())


if __name__ == '__main__':
    main()
