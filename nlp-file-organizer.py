from nltk import everygrams
from nltk.corpus import wordnet as wn
import spacy
import os
from collections import deque, Counter
import numpy as np
from scipy.cluster.vq import kmeans2
import sys
import itertools

parent_directory = sys.argv[1]
file_names = [a_file for a_file in os.listdir(parent_directory) if os.path.isfile(os.path.join(parent_directory, a_file))]
nlp = spacy.load('en')
words_representatives = deque()
feature_vecs = list()

print("Extracting words...")
# Extract words by making in-order permutations of word characters and determining if they're words
for file_name in file_names:
    name = os.path.splitext(file_name)[0].lower()
    wordlist = [''.join(x) for x in everygrams(name, min_len=3) if len(wn.synsets(''.join(x))) > 0]

    # Backup - try the spacy web corpus if we don't find anything from wordnet
    if len(wordlist) == 0:
        wordlist = [''.join(x) for x in everygrams(name, min_len=3) if ''.join(x) in nlp.vocab]

    wordlist = wordlist[len(wordlist) - 5:len(wordlist)]
    print("{0}: {1}".format(name, wordlist))
    mydoc = nlp(' '.join(wordlist))
    words_representatives.append((file_name, wordlist))
    feature_vecs.append(mydoc.vector)

print("Clustering...")
# Cluster with the K-Means algorithm
feature_vecs_numpy = np.array(feature_vecs)
num_clusters = int(sys.argv[2])
clusters = list()

for member in range(0, num_clusters):
    clusters.append(deque())

print("Attempting to find {0} clusters.".format(num_clusters))
kmeans_result = kmeans2(feature_vecs_numpy, num_clusters)
cluster_assignments = kmeans_result[1]

for index in range(0, len(words_representatives)):
    clusters[cluster_assignments[index]].append(words_representatives[index])

print("Clustered data:")
cluster_names = deque()

# Find names of cluster by either a common hypernym of the word list or the longest, most frequent term
for cluster in clusters:
    representatives = list(itertools.chain(*[x[1] for x in cluster]))
    names = [x[0] for x in cluster]

    if len(representatives) > 0:
        hypernym = wn.synsets(representatives[0])
        if len(hypernym) > 0:
            hypernym = hypernym[0]
        else:
            hypernym = ""

        for index in range(1, len(representatives)):
            lookup = wn.synsets(representatives[index])

            if len(lookup) > 0:
                if not isinstance(hypernym, str):
                    hypernym = lookup[0].lowest_common_hypernyms(hypernym)

                    if len(hypernym) > 0:
                        hypernym = hypernym[0]
                    else:
                        hypernym = lookup[0]
                else:
                    hypernym = lookup[0]
            
        if not isinstance(hypernym, str):
            rep_word = hypernym.lemmas()[0].name()
        else:
            rep_word = ""

        if rep_word == "" or hypernym.min_depth() < 2:
            rep_word = Counter(representatives).most_common(5)
            longest_most_common = ""

            for word in rep_word:
                if len(word[0]) > len(longest_most_common):
                    longest_most_common = word[0]

            rep_word = longest_most_common

        cluster_names.append(rep_word)
        print("{0}: {1}".format(rep_word, names))
    else:
        cluster_names.append("EMPTY-CLUSTER") 

print("Moving files to clustered directories...")
# Copy the files to the appropriate directories
for index in range(0, len(clusters)):
    folder_name = cluster_names[index]

    if folder_name != "EMPTY-CLUSTER":
        full_folder_path = os.path.join(parent_directory, folder_name)

        if not os.path.isdir(full_folder_path):
            os.mkdir(full_folder_path)
        
        for file_name in clusters[index]:
            old_file_path = os.path.join(parent_directory, file_name[0])
            new_file_path = os.path.join(full_folder_path, file_name[0])
            os.rename(old_file_path, new_file_path)

# Put all the stragglers in an UNCATEGORIZED directory
remaining_files = [file_name for file_name in os.listdir(parent_directory) \
    if os.path.isfile(os.path.join(parent_directory, file_name))]

if len(remaining_files) > 0:
    uncategorized_dir = os.path.join(parent_directory, "UNCATEGORIZED")
    if not os.path.isdir(uncategorized_dir):
        os.mkdir(uncategorized_dir)

    for file_name in remaining_files:
        old_file_path = os.path.join(parent_directory, file_name)
        new_file_path = os.path.join(uncategorized_dir, file_name)
        os.rename(old_file_path, new_file_path)


print("Done!")