from transformers import AutoTokenizer, TFAutoModel
import tensorflow as tf
import numpy as np
from preprocessing import get_report_labels

SEQ_LEN = 128

class MSNR(tf.keras.Model):
    def __init__(self):
        super(MSNR, self).__init__()

        # Initializa model and tokenizer
        file_name = "giacomomiolo/biobert_reupload"
        self.biobert_tokenizer = AutoTokenizer.from_pretrained(file_name)
        self.biobert = TFAutoModel.from_pretrained(file_name)
        self.labels = get_report_labels()
        self.num_examples = len(self.labels)
        self.input_ids = np.zeros((self.num_examples, SEQ_LEN))
        self.input_mask = 

    
    def tokenize_sentences(self, sentence):
        """
            Input: sentence as string
            Output: tokenized sentence array (including special tokens) and attention mask as array (1s for word and 0 o.w.)
        """
        tokens = self.biobert_tokenizer.encode_plus(sentence, max_length=SEQ_LEN,
                                   truncation=True, padding='max_length',
                                   add_special_tokens=True, return_attention_mask=True,
                                   return_token_type_ids=False, return_tensors='tf')

        return tokens['input_ids'], tokens['attention_mask']

    def one_hot_encode_labels(self):
        one_hot_labels = np.zeros((len(self.labels), 7))
        pass
        
    def call(self, inputs):   
        embeddings = self.get_embeddings(inputs)

        sentenceEmbeddings = self.encoder(embeddings)

        aggregate_embeddings = self.layer2(sentenceEmbeddings)
        
        prbs = self.dense(aggregate_embeddings)
        
    
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