from transformers import AutoTokenizer, TFAutoModel, file_utils
import tensorflow as tf
import numpy as np
from preprocessing import get_data
import random

SEQ_LEN = 128

class MSNR():
    def __init__(self):
        # super(MSNR, self).__init__()

        # Initializa model and tokenizer
        file_name = "giacomomiolo/biobert_reupload"
        # file_name = "bert-base-cased"
        self.biobert_tokenizer = AutoTokenizer.from_pretrained(file_name)
        self.biobert = TFAutoModel.from_pretrained(file_name)

        # class variables
        self.num_labels = 6
        self.impressions, self.labels = get_data()

        newImpressions = []
        newLabels = []

        for i, thing in enumerate(self.labels):
            if thing != 7:
                newImpressions.append(self.impressions[i])
                newLabels.append(self.labels[i])

        self.impressions = newImpressions
        self.labels = newLabels


        # dictG = {}
        # for b in self.labels:
        #     if b not in dictG:
        #         dictG[b] = 1
        #     else:
        #         dictG[b] += 1
        # print(dictG)

        tog = list(zip(self.impressions, self.labels))
        random.shuffle(tog)
        self.impressions, self.labels = zip(*tog)

        self.endImp = self.impressions[-160:]
        self.impressions = self.impressions[:-160]

        self.endLab = self.labels[-160:]
        self.labels = self.labels[:-160]

        # Input and layers
        # self.dense_layer = tf.keras.layers.Dense(self.num_labels, activation='softmax', name='outputs')  # adjust based on number of sentiment classes
        self.input_ids = tf.keras.layers.Input(shape=(SEQ_LEN,), name='input_ids', dtype='int32')
        self.mask = tf.keras.layers.Input(shape=(SEQ_LEN,), name='attention_mask', dtype='int32')

        # Model variables
        self.optimizer = tf.keras.optimizers.Adam(0.00005)
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
        dataset = dataset.shuffle(1000).batch(8)

        # create train and test sets
        # train_data = dataset.take(round(len(dataset) * 0.8))
        # test_data = dataset.skip(round(len(dataset) * 0.8))

        train_data = dataset
        test_data = None

        # # free space
        # del dataset

        return self.biobert(self.input_ids, attention_mask=self.mask)[0], train_data, test_data, self.mask

    def map_func(self, input_ids, masks, labels):
        """
            Purpose: structure data for input to BERT
            Input: input_id, mask and encoded label arrays from our impressions and annotations
            Output: tuple of (dictionary mapping name to array, labels)
        """
        return {'input_ids': input_ids, 'attention_mask': masks}, labels

    def predict_impression(self, impression):
        ids, mask = self.tokenize(impression)
        return self.model.predict([ids, mask])


    def train_model(self):
        """
            Purpose: Instantiates and trains the model
            Input: None
            Output: None
        """
        embeddings, train_data, test_data, mask = self.get_biobert_embeddings()
        # outputs = self.dense_layer(embeddings)
        X = tf.keras.layers.GlobalAveragePooling1D()(embeddings, mask)  # reduce tensor dimensionality
        # X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Dense(768)(X)
        X=  tf.keras.layers.ThresholdedReLU(theta=0.1)(X)
        X = tf.keras.layers.Dropout(0.75)(X)
        y = tf.keras.layers.Dense(6, activation='softmax', name='outputs')(X)
        self.model = tf.keras.Model(inputs=[self.input_ids, self.mask], outputs=y)

        # freeze the BERT layer
        self.model.layers[2].trainable = False

        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=[self.accuracy])

        # and train it



        history = self.model.fit(train_data, epochs=175)
        print(history)


        correct= 0
        total = len(self.endImp)
        labelMap = {}
        for i in range(1,7):
            labelMap[i] = [0.0,0.0]

        for i in range(len(self.endImp)):


            prediction = self.predict_impression(self.endImp[i])

            if tf.math.argmax(prediction[0]).numpy() + 1 == self.endLab[i]:
                correct += 1
                labelMap[self.endLab[i]][0] += 1
            labelMap[self.endLab[i]][1] += 1
        print('\n')
        print(correct/total)
        print('\n')
        for thing in labelMap:
            if labelMap[thing][1] != 0:
                print(str(thing) + " : " + str(labelMap[thing][0]/labelMap[thing][1]))
            else:
                print(str(thing) + " : " + "N/A")
            print(labelMap[thing][1])
            print(" ")

        # results = self.model.evaluate(test_data)
        #
        # print("results", results)
        #
        # print('\n')
        # for i in range(20):
        #
        #     print(self.predict_impression(self.endImp[i]))
        #     print(self.endLab[i])
        #     print('\n')




def main():

    m = MSNR()
    print(m.train_model())
    # print(m.predict_impression("1.  No definite or significant interval change in the size of the abnormal soft tissue density within the anterior left ethmoid air cells, consistent with residual tumor and/or residual changes at site of tumor.  There is new ethmoidal mucosal sinus disease, which makes differentiating residual tumor from surrounding sinus disease difficult.  No frank intracranial extension is identified, though there appears to be unchanged slight intraorbital extension. Further follow-up suggested. 2.  New ethmoidal and maxillary sinus disease, with interval development of air-fluid level within the left maxillary sinus, suggesting acute sinusitis."))


if __name__ == '__main__':
    main()
