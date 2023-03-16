#  NMT Model

English-to-German neural machine translation (NMT) model using Long Short-Term Memory (LSTM) networks with attention
Details:


learn how to preprocess your training and evaluation data using Opus dataset (opus/medical which has medical related texts)
implement an encoder-decoder system with attention
Implement attention using Scaled Dot Product Attention
build the NMT model from scratch using Trax
generate translations using greedy and Minimum Bayes Risk (MBR) decoding


## RNN
- Recurrent Neural Network (RNN) with LSTMs can work for short to medium length sentences but can result in vanishing gradients for very long sequences. All of the context of the input sentence is compressed into one vector that is passed into the decoder block, creating a bottleneck.  Attention mechanism to allow the decoder to access all relevant parts of the input sentence regardless of its length.
-

![Alt text](image/seq2seq.png "Seq2seq overview")


![Alt text](image/attention.png "Attention mechanism")
#

Tokenization: Tokenization is cutting input data into meaningful parts that can be embedded into a vector space.

Embedding: Want to represent data as numbers to compute our tasks. Embeddings contain semantic information about the word.
Example: Word2Vec is an algotihm that use neural network to learn word association published in 2013 and developed by Tomas Mikolov at Google. The semantic similarity is defined by the consine similarity between the two vectors. Each word in your vocabulary is represented by a one-hot encoded vector. Vectors have tipically between 100 - 300 dimensions.

Bucketing: Bucketing the tokenized sentences is an important technique used to speed up training in NLP. It consists to group the tokenized sentences by length and bucket, this way we only add minimal padding and save computation resources

## Score

BlEU Score (Bilingual Evaluation Understudy): between 0 and 1, closer to one beging better, compare candidates translations to reference (human) translations. It is based on correct words count. However, it doesn't consider semantic meaning or sentence structure

Rouge-N Score (Recall-Oriented Understudy of Gisting Evaluation):

Bleu mesure precision and rouge measure recall. Bleu measures how much the words (and/or n-grams) in the machine generated summaries appeared in the human reference summaries. Rouge measures how much the words (and/or n-grams) in the human reference summaries appeared in the machine generated summaries. Both of them defines the F1 score using the formula: 
$$ F1 = 2 * (Bleu * Rouge) / (Bleu + Rouge) $$


Brevety penalty

## Temperature

Temperature can control for more or lessrandomness in predictions. lower temperature setting means more condifent and more conservative network.

Beam search decoding. The most probable translation is not composed of the most probable word at each step. The solution is to calculate the probability of multiple possible sequences by doing beam search. Beam width B determines the number of sequences you keep. It penalizes long sequences so the sequence length should be normalized

Minimum Bayes risk: it generates several candidates translation 

## Evaluation

The siamese network is evaluated using accuracy, depending on a threshold t. The threshold selected is 0.7' A higher threshold means that only very similar questions will be considered as the same question

Accuracy increases with the threshold unitl a threshold above 0.7, the accuracy decreases

![Alt text](image/accuracy.png "Accuracy")

## Resources
- [Attention is all your need](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
- [Natural Language Processing with Attention Models](https://www.coursera.org/learn/attention-models-in-nlp/home/week/1)
