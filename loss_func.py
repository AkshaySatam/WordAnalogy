import tensorflow as tf
import numpy as np
import math as m
def cross_entropy_loss(inputs, true_w):
    """
    ==========================================================================

    inputs: The embeddings for context words. Dimension is [batch_size, embedding_size].
    true_w: The embeddings for predicting words. Dimension of true_w is [batch_size, embedding_size].

    Write the code that calculate A = log(exp({u_o}^T v_c))
    Will calculate A here
    """
    """"
    y=0;

    # I have calculated the below for all v_c. This may not be true.
    for j in true_w:
        for i in inputs:
            x = 0;
            x += m.exp(tf.transpose(inputs[i]) * true_w[j]);
        # This could be wrong. need to check on this.
        y += m.log(x);
    B = m.log(y);
    """

    #Need to rethink on this.
    A = tf.reduce_sum(np.multiply(true_w,inputs),1,keepdims=True)

    trns = tf.transpose(true_w)
    mult = tf.matmul(inputs,trns)
    mat_sum = tf.reduce_sum(tf.exp(mult), 1, keepdims=True)
    B = tf.log(mat_sum)
    """
    And write the code that calculate B = log(\sum{exp({u_w}^T v_c)})


    B =

    ==========================================================================
    """
    return tf.subtract(B, A)

def nce_loss(inputs, weights, biases, labels, sample, unigram_prob):
    """
    ==========================================================================

    inputs: Embeddings for context words. Dimension is [batch_size, embedding_size].
    weigths: Weights for nce loss. Dimension is [Vocabulary, embeeding_size].
    biases: Biases for nce loss. Dimension is [Vocabulary, 1].
    labels: Word_ids for predicting words. Dimesion is [batch_size, 1]. - This is the list of predicted words
    samples: Word_ids for negative samples. Dimension is [num_sampled].
    unigram_prob: Unigram probability. Dimesion is [Vocabulary].

    Implement Noise Contrastive Estimation Loss Here

    ==========================================================================
    """
    #Get label embeddings
    k = tf.size(sample)
    label_embeddings = tf.nn.embedding_lookup(weights,labels)
    label_embeddings = tf.reshape(label_embeddings, [-1, 128])
    #matmul with transpose.
    score = tf.reduce_sum(np.multiply(label_embeddings, inputs), 1, keepdims=True)
    probs = tf.convert_to_tensor(unigram_prob, dtype=tf.float32)
    probs_vector = tf.nn.embedding_lookup(probs,labels)
    bias_vector = tf.nn.embedding_lookup(biases, labels)
    print(score, "Shape of score")
    print(bias_vector, "Shape of bias_vector")
    print(probs_vector, "Shape of probs_vector")
    x = tf.subtract (tf.add(score,bias_vector) , tf.log(tf.add(1e-10,tf.scalar_mul(k,probs_vector))))
    print(x, "Shape of x")
    """
    minux_x = tf.scalar_mul(-1,x)
    exp_x = tf.exp(minux_x)
    add_1_to_x = tf.add(1.0,x)
    A = tf.log(tf.add(1e-10,tf.reciprocal(tf.sigmoid(x))))
    """
    A = tf.log(tf.add(1e-10,tf.sigmoid(x)))

    sample_embeddings = tf.nn.embedding_lookup(weights,sample)
    sample_embeddings_transpose = tf.transpose(sample_embeddings)
    #input transpose and vice versa
    prod = tf.matmul(inputs,sample_embeddings_transpose)
    #prodT = tf.transpose(prod)
    sample_biases = tf.nn.embedding_lookup(biases, sample)
    sample_biasesTr = tf.transpose(sample_biases)
    addBias = tf.add(prod,sample_biasesTr)
    print(addBias, "Shape of addBias")
    probsSample = tf.nn.embedding_lookup(probs,sample)
    print(probsSample, "Shape of probs Sample")
    y = tf.subtract(addBias,tf.log(tf.add(1e-10,tf.scalar_mul(k,tf.transpose(probsSample)))))
    """
    minux_y = tf.scalar_mul(-1, y)
    exp_y = tf.exp(minux_y)
    add_1_to_y = tf.add(1.0, y)
    inverse_add_1 = tf.reciprocal(add_1_to_y)
    minus_add1Y = tf.add(1.0,tf.scalar_mul(-1,inverse_add_1))
    """
    minus_add1Y = tf.add(1.0, tf.scalar_mul(-1, tf.sigmoid(y)))
    B = tf.reduce_sum(tf.log(tf.add(1e-10,minus_add1Y)),1,keepdims=True)
    print(B, "Shape of B")
    print(A, "Shape of A")
    return tf.scalar_mul(-1,tf.add(A,B))