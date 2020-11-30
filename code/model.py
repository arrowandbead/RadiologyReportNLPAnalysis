from transformers import AutoTokenizer, TFAutoModel
import tensorflow as tf
import numpy as np

class MSNR(tf.keras.Model):
	def __init__(self, vocab_size):
		super(MSNR, self).__init__()
        
        self.embeddings_tokenizer = AutoTokenizer.from_pretrained("giacomomiolo/biobert_reupload")

        self.embeddings_model = TFAutoModel.from_pretrained("giacomomiolo/biobert_reupload")

        self.batch_size = 10 #placeholder
        self.embedding_size = np.shape(self.get_embeddings)
        self.vocab_size = 0 # placeholder

        # self.bioBert_embedding = tf.Variable(tf.random.normal(shape=[self.vocab_size, self.embedding_size], stddev=.01, dtype=tf.float32))

        # Define encoder and decoder layers:
        # self.encoder = transformer.Transformer_Block(self.embedding_size, is_decoder=False, multi_headed=False)
        # self.decoder = transformer.Transformer_Block(self.embedding_size, is_decoder=True, multi_headed=False)
        
        #creating encoder layer: to create embeddings for each of the sentences
        self.encoder = tf.keras.layers.Transformer_Block(self.embedding_size, is_decoder=False, multi_headed=False)

        self.block= tf.keras.layers.transformer.Transformer_Block
        self.tBlock2 = tf.keras.layers.Transformer_Block(self.embedding_size)
        # Define dense layer(s)
        self.dense_layer = tf.keras.layers.Dense(self.vocab_size, activation='softmax')
    
    def call(self, inputs):   
        embeddings = self.get_embeddings(inputs)

        sentenceEmbeddings = self.encoder(embeddings)

        aggregate_embeddings = self.layer2(sentenceEmbeddings)
        
        prbs = self.dense(aggregate_embeddings)
        
    
    def get_embeddings(self):
        batch = tokenizer(
            ["Moderate interval decrease in size of mass located about the anterior aspect of the left parotid gland when compared to prior CT examination.", 
            "Internal architecture of this lesion demonstrates cystic changes which is presumably post therapeutic."],
            padding=True,
            truncation=True,
            return_tensors="tf"
            )
        outputs = self.embedding_model(batch, output_hidden_states=True)
        hidden_states = outputs[2]
        embeddings = hidden_states[0]

        return embeddings


def main():
    # TODO: Return the training and testing data from get_data
    
    # TODO: Instantiate model
    model = Model()

    # TODO: Train and test for up to 15 epochs.
    '''
    num_epoch = 15
    for epoch in range(num_epoch):
        print('Epoch {}: '.format(epoch))
        train(model, train_data)
    pass

    acc = test(model, test_data)
    print('Test Accuracy =', format(acc * 100))
    '''
if __name__ == '__main__':
    main()
