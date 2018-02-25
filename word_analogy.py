import os
import pickle
import numpy as np


def getDifference(a,b):
    #Code referenced from : https://masongallo.github.io/machine/learning,/python/2016/07/29/cosine-similarity.html

    #return cosine difference
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

def writeOutput(outputPairs, maxIndex, minIndex):
    with open('word_analogy_dev_predictions.txt', 'a') as the_file:
        for o in outputPairs:
            the_file.write(o+' ')
        the_file.write(outputPairs[maxIndex]+' '+outputPairs[minIndex]+'\n')

model_path = './models/'
#loss_model = 'cross_entropy'
loss_model = 'nce'

model_filepath = os.path.join(model_path, 'word2vec_%s.model'%(loss_model))

dictionary, steps, embeddings = pickle.load(open(model_filepath, 'rb'))

"""
==========================================================================

Write code to evaluate a relation between pairs of words.
You can access your trained model via dictionary and embeddings.
dictionary[word] will give you word_id
and embeddings[word_id] will return the embedding for that word.

word_id = dictionary[word]
v1 = embeddings[word_id]

or simply

v1 = embeddings[dictionary[word_id]]

==========================================================================
"""

print(dictionary['sobbing'],"Sobbing's ID")
print(embeddings[dictionary['sobbing']],"Sobbing's embedding")

with open('word_analogy_dev.txt', 'r') as f:
    for line in f:
        count_relationPairs = 0
        avg_relationsPairs = 0.0
        difference_relationsPairs = 0.0
        maxDiff = -float("inf")
        minDiff = float("inf")
        maxIndex = minIndex = -1
        count_optionPairs = 0

        #Tokenize the line
        splittedLine = line.split('||')
        relations = splittedLine[0]
        if(splittedLine[1][-1:]) == '\n':
            options = splittedLine[1][0:-1]
        else:
            options = splittedLine[1]
        """
        print(relations,"Relations array")
        print(options, "options array")
        """
        relations_pairs = relations.split(',')
        options_pairs = options.split(',')

        print(relations_pairs, "relations_pairs array")
        print(options_pairs, "options_pairs array")

        for r in relations_pairs:
            indRelation = r[1:-1].split(':')
            #calculate the mean difference
            """
            print(indRelation[0],"Rel 0")
            print(indRelation[1], "Rel 1")
            """
            difference_relationsPairs += getDifference(embeddings[dictionary[indRelation[0]]],embeddings[dictionary[indRelation[1]]])
            count_relationPairs+=1
        avg_relationsPairs = difference_relationsPairs/count_relationPairs
        for o in options_pairs:
            indOption = o[1:-1].split(':')
            """
            print(indOption[0], "Opt 0")
            print(indOption[1], "Opt 1")
            """
            diff = getDifference(embeddings[dictionary[indOption[0]]],embeddings[dictionary[indOption[1]]])
            if abs(diff-avg_relationsPairs) > maxDiff:
                maxDiff = abs(diff-avg_relationsPairs)
                maxIndex= count_optionPairs
            if abs(diff-avg_relationsPairs) < minDiff:
                minDiff = abs(diff-avg_relationsPairs)
                minIndex= count_optionPairs
            count_optionPairs+=1
        print(minIndex, "Min Index")
        print(maxIndex, "Max Index")
        print(options_pairs[minIndex], "Most matching pair")
        print(options_pairs[maxIndex], "Least matching pair")
        #write the line to o/p file in the specified format
        writeOutput(options_pairs,maxIndex,minIndex)


