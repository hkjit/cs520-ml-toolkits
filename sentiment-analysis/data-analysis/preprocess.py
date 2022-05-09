import pandas as pd
import numpy as np
from numpy import savetxt
from collections import Counter

df = pd.read_csv("./data/Tweets.csv")
# removing duplicates
df.drop_duplicates(keep='first',inplace=True)

# remove unnecessay punctuations
punctuations_not_needed = '!"#$%&\'()*+,-./:;<=>?[\\]^_`{|}~'
reviews = np.array(df['text'])[:14640]
labels = np.array(df['airline_sentiment'])[:14640]
for i in range(len(reviews)):
    review = reviews[i].lower()
    reviews[i] = ''.join([c for c in review if c not in punctuations_not_needed])

reviews = np.array(df['text'])[0:14000]
labels = np.array(df['airline_sentiment'])[0:14000]
all_text = ' '.join(reviews)

# create a list of words
words = all_text.split()

# get rid of web address, twitter id, and digit
new_reviews = []
for review in reviews:
    review = review.split()
    new_text = []
    for word in review:
        if (word[0] != '@') & ('http' not in word) & (~word.isdigit()):
            new_text.append(word)
    new_reviews.append(new_text)

# Build a dictionary that maps words to integers
counts = Counter(words)
vocab = sorted(counts, key=counts.get, reverse=True)
vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}
print(len(vocab_to_int))

# use the dict to tokenize each review in reviews_split
# store the tokenized reviews in reviews_ints
reviews_ints = []
for review in new_reviews:
    reviews_ints.append([vocab_to_int[word] for word in review])

# 1=positive, 1=neutral, 0=negative label conversion
encoded_labels = []
for label in labels:
    if label == 'neutral':
        encoded_labels.append(1)
    elif label == 'negative':
        encoded_labels.append(0)
    else:
        encoded_labels.append(1)

encoded_labels = np.asarray(encoded_labels)

# Dataset parameters
split_ratio = 0.8
seq_length = 30
batch_size = 50
# getting the correct rows x cols shape
features = np.zeros((len(reviews_ints), seq_length), dtype=int)

# for each review, I grab that review and
for i, row in enumerate(reviews_ints):
    features[i, -len(row):] = np.array(row)[:seq_length]


# split data into training, validation, and test data (features and labels, x and y)
split_idx = int(len(features) * split_ratio)
train_x, remaining_x = features[:split_idx], features[split_idx:]
train_y, remaining_y = encoded_labels[:split_idx], encoded_labels[split_idx:]

test_idx = int(len(remaining_x) * 0.5)
val_x, test_x = remaining_x[:test_idx], remaining_x[test_idx:]
val_y, test_y = remaining_y[:test_idx], remaining_y[test_idx:]

# save to csv file
savetxt('./data/preprocessed_training_inputs.csv', train_x, delimiter=',')
savetxt('./data/preprocessed_training_labels.csv', train_y, delimiter=',')
savetxt('./data/preprocessed_validation_inputs.csv', val_x, delimiter=',')
savetxt('./data/preprocessed_validation_labels.csv', val_y, delimiter=',')
savetxt('./data/preprocessed_testing_inputs.csv', test_x, delimiter=',')
savetxt('./data/preprocessed_testing_labels.csv', test_y, delimiter=',')

