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

        # Initialize model and tokenizer
        file_name = "giacomomiolo/biobert_reupload"
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

        # newImpressions = []
        # newLabels = []

        # for i, thing in enumerate(self.labels):
        #     if thing != 7:
        #         newImpressions.append(self.impressions[i])
        #         newLabels.append(self.labels[i])

        # self.mimpressions = newImpressions
        # self.mlabels = newLabels

        # tog = list(zip(self.mimpressions, self.mlabels))
        # random.shuffle(tog)
        # self.mimpressions, self.mlabels = zip(*tog)

        # self.mendImp = self.mimpressions[-160:]
        # self.mimpressions = self.mimpressions[:-160]

        # self.mendLab = self.labels[-160:]
        # self.mlabels = self.mlabels[:-160]
        
    class AccuracyCallback(tf.keras.callbacks.Callback):
        def __init__(self, x, y, model):
            self.x = x
            self.y = y
            self.y_true = np.array([])

            # decode one-hot labels in each batch
            for batch in y:
                print("Y batch:", batch)
                batch_labels_as_array = tf.make_ndarray(tf.make_tensor_proto(batch))
                self.y_true = np.append(self.y_true, np.argmax(batch_labels_as_array, axis=1))
            
            # for batch in x:
            #     print("X batch:", batch)
            # self.reports = []

        def on_epoch_end(self, epoch, logs={}):
            # predictions = []
            # for batch in self.x:
            #     for example in batch:
            #         y_predicted = self.model.predict(example)
            #         print("prediction:", y_predicted)
            #         predictions.append(int(y_predicted))    

            # true_label = []
            # for batch in self.y:
            #     for example_label in y:
            #         true_label.append(int(example_label))
                
            y_predicted = np.argmax(np.asarray(self.model.predict(self.x)), axis=1)
            print("y predicted:", y_predicted)
            # report = classification_report(self.y_true, y_predicted, labels=[0, 1, 2, 3, 4, 5, 6], output_dict=True)
            # self.reports.append(report)


            correct_examples = np.zeros(7)
            total_examples = np.zeros(7)
            accuracies = np.zeros(7)
            overall_epoch_accuracy = np.mean(np.where(self.y_true == y_predicted, 1, 0))
            # overall_epoch_accuracy = np.mean(np.where(y_true == predictions, 1, 0))

            for i in range(len(self.y_true)):
                total_examples[int(self.y_true[i])] += 1
                correct_examples[int(self.y_true[i])] += 1 if int(self.y_true[i]) == int(y_predicted[i]) else 0 
            
            for i in range(7):
                accuracies[i] = correct_examples[i] / total_examples[i] if total_examples[i] != 0 else 0.0
            # print("\n")
            # print("accuracy per class:", [report[str(label)]['recall'] for label in range(7)]) # recall = TP / (TP + FN)
            # print("number of examples per class:", [report[str(label)]['support'] for label in range(7)])
            # print("\n")
            print("\n")
            print("accuracy per class:", accuracies)
            print("# of examples per class:", total_examples)
            print("epoch accuracy as calculated by callback:", overall_epoch_accuracy)
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
        dataset = dataset.shuffle(len(self.impressions)).batch(32) # this buffer size should perfectly shuffle

        # create train and test sets
        train_data = dataset.take(round(len(dataset) * 0.8))
        test_data = dataset.skip(round(len(dataset) * 0.8))

        len(train_data)
        len(test_data)

        # free space
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
            Input: {input_id:mask} dictionary and encoded labels tensor
            Output: labels tensor
        """
        return labels
  
    def predict_impression(self, impression):
        ids, mask = self.tokenize(impression)
        return self.model.predict([ids, mask])

    def run_model(self):   
        """
            Purpose: Instantiates and trains the model
            Input: None
            Output: None
        """
        embeddings, train_data, test_data = self.get_biobert_embeddings()
        X = tf.keras.layers.GlobalMaxPool1D()(embeddings)  # reduce tensor dimensionality
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Dense(128, activation='relu')(X)
        X = tf.keras.layers.Dropout(0.1)(X)
        y = tf.keras.layers.Dense(7, activation='softmax', name='outputs')(X) 
       
        # X = tf.keras.layers.GlobalAveragePooling1D()(embeddings)  # reduce tensor dimensionality
        # X = tf.keras.layers.BatchNormalization()(X)
        # X = tf.keras.layers.Dense(768)(X)
        # X=  tf.keras.layers.ThresholdedReLU(theta=0.1)(X)
        # X = tf.keras.layers.Dropout(0.75)(X)
        # y = tf.keras.layers.Dense(7, activation='softmax', name='outputs')(X)
        
        model = tf.keras.Model(inputs=[self.input_ids, self.mask], outputs=y)

        # freeze the BERT layer
        model.layers[2].trainable = False

        per_class_accuracy_train = MSNR.AccuracyCallback(train_data.map(self.get_input_ids_and_mask), train_data.map(self.get_labels), model)

        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=[self.accuracy])


        # and train it
        print("-" * 15, "TRAIN RESULTS", "-" * 15)
        # model.fit(train_data, epochs=175, callbacks=[per_class_accuracy_train])
        model.fit(train_data, epochs=20, callbacks=[per_class_accuracy_train]) 
 
        print("-" * 55)

        print("-" * 15, "TEST RESULTS", "-" * 15)
        results = model.evaluate(test_data)
        print(results)
        print("-" * 55)

        test_ids_and_mask = test_data.map(self.get_input_ids_and_mask)
        test_labels = test_data.map(self.get_labels)

        # get per class accuracy
        per_class_accuracy_test = MSNR.AccuracyCallback(test_data.map(self.get_input_ids_and_mask), test_data.map(self.get_labels), model)
        per_class_accuracy_test.on_epoch_end(0)

        y_true = np.array([])

        # check
        for batch_of_labels in test_labels:
                batch_labels_as_array = tf.make_ndarray(tf.make_tensor_proto(batch_of_labels))
                y_true = np.append(y_true, np.argmax(batch_labels_as_array, axis=1))

        y_predicted = np.argmax(np.asarray(self.model.predict(test_ids_and_mask)), axis=1)
        correct_examples = np.zeros(7)
        total_examples = np.zeros(7)
        accuracies = np.zeros(7)
        overall_epoch_accuracy = np.mean(np.where(y_true == y_predicted, 1, 0))

        for i in range(len(y_true)):
            total_examples[int(y_true[i])] += 1
            correct_examples[int(y_true[i])] += 1 if int(y_true[i]) == int(y_predicted[i]) else 0 
            
        for i in range(7):
            accuracies[i] = correct_examples[i] / total_examples[i] if total_examples[i] != 0 else 0.0
            # print("\n")

        # correct = 0
        # total = 7
        # labelMap = {}
        # for i in range(1,7):
        #     labelMap[i] = [0.0,0.0]

        # for i in range(len(self.endImp)):

        #     # prediction = self.predict_impression(self.endImp[i])

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
    m.run_model()

if __name__ == '__main__':
    main()
