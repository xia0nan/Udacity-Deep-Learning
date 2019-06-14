# TODO: -Copy the SentimentNetwork class from Project 5 lesson
#       -Modify it according to the above instructions 
import time
import sys
import numpy as np

# Encapsulate our neural network in a class


class SentimentNetwork:
    def __init__(self, reviews, labels, hidden_nodes=10, min_count=50, polarity_cutoff=0.1, learning_rate=0.1):
        """Create a SentimenNetwork with the given settings
        Args:
            reviews(list) - List of reviews used for training
            labels(list) - List of POSITIVE/NEGATIVE labels associated with the given reviews
            hidden_nodes(int) - Number of nodes to create in the hidden layer
            learning_rate(float) - Learning rate to use while training
        
        """
        # Assign a seed to our random number generator to ensure we get
        # reproducable results during development
        np.random.seed(1)

        self.min_count = min_count
        self.polarity_cutoff = polarity_cutoff

        # process the reviews and their associated labels so that everything
        # is ready for training
        self.pre_process_data(reviews, labels)

        # Build the network to have the number of hidden nodes and the learning rate that
        # were passed into this initializer. Make the same number of input nodes as
        # there are vocabulary words and create a single output node.
        self.init_network(len(self.review_vocab),
                          hidden_nodes, 1, learning_rate)

    def get_review_vocab(self, reviews, labels):
        """ Return vocab that meets min_count and polarity_cutoff requirement
        """
        total_counts = Counter()
        pos_neg_ratios = Counter()

        review_vocab = set()

        for i in range(len(reviews)):
            word_list = reviews[i].split(' ')
            if labels[i] == "POSITIVE":
                positive_counts.update(word_list)
            elif labels[i] == "NEGATIVE":
                negative_counts.update(word_list)

            total_counts.update(word_list)

        for (word, cnt) in total_counts.most_common():
            if cnt > self.min_count:
                pos_neg_ratios[word] = positive_counts[word] / float(negative_counts[word]+1)
                # map to polar axis
                if(pos_neg_ratios[word] > 1):
                    pos_neg_ratios[word] = np.log(pos_neg_ratios[word])
                else:
                    pos_neg_ratios[word] = -np.log((1 / (pos_neg_ratios[word] + 0.01)))

                if np.abs(pos_neg_ratios[word]) >= self.polarity_cutoff:
                    review_vocab.add(word)
        
        return review_vocab
        

    def pre_process_data(self, reviews, labels):

        review_vocab = set()

        # words are only added to the vocabulary if 
        #   they occur in the vocabulary more than min_count times
        # words are only added to the vocabulary if the absolute value of 
        #   their postive-to-negative ratio is at least polarity_cutoff
        review_vocab = self.get_review_vocab(reviews, labels)

        # Convert the vocabulary set to a list so we can access words via indices
        self.review_vocab = list(review_vocab)

        label_vocab = set()
        # TODO: populate label_vocab with all of the words in the given labels.
        #       There is no need to split the labels because each one is a single word.
        for label in labels:
            label_vocab.add(label)

        # Convert the label vocabulary set to a list so we can access labels via indices
        self.label_vocab = list(label_vocab)

        # Store the sizes of the review and label vocabularies.
        self.review_vocab_size = len(self.review_vocab)
        self.label_vocab_size = len(self.label_vocab)

        # Create a dictionary of words in the vocabulary mapped to index positions
        self.word2index = {}
        # TODO: populate self.word2index with indices for all the words in self.review_vocab
        #       like you saw earlier in the notebook
        for i, word in enumerate(self.review_vocab):
            self.word2index[word] = i

        # Create a dictionary of labels mapped to index positions
        self.label2index = {}
        # TODO: do the same thing you did for self.word2index and self.review_vocab,
        #       but for self.label2index and self.label_vocab instead
        for i, label in enumerate(self.label_vocab):
            self.label2index[label] = i

    def init_network(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Store the number of nodes in input, hidden, and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # print("input_nodes", input_nodes)
        # print("hidden_nodes", hidden_nodes)
        # print("output_nodes", output_nodes)

        # Store the learning rate
        self.learning_rate = learning_rate

        # Initialize weights

        # TODO: initialize self.weights_0_1 as a matrix of zeros. These are the weights between
        #       the input layer and the hidden layer.
        self.weights_0_1 = np.zeros(shape=(input_nodes, hidden_nodes))

        # TODO: initialize self.weights_1_2 as a matrix of random values.
        #       These are the weights between the hidden layer and the output layer.
        self.weights_1_2 = np.random.normal(
            0, scale=hidden_nodes**-0.5, size=(hidden_nodes, output_nodes))

        # Layer 1: Hidden Layer, Shape = 1 x num_hidden
        self.layer_1 = np.zeros(shape=(1, hidden_nodes))

    def get_target_for_label(self, label):
        # TODO: Copy the code you wrote for get_target_for_label
        #       earlier in this notebook.
        return 1 if label == "POSITIVE" else 0

    def sigmoid(self, x):
        # TODO: Return the result of calculating the sigmoid activation function
        #       shown in the lectures
        return 1 / (1 + np.exp(-x))

    def sigmoid_output_2_derivative(self, output):
        # TODO: Return the derivative of the sigmoid activation function,
        #       where "output" is the original output from the sigmoid fucntion
        return output * (1 - output)

    def train(self, training_reviews_raw, training_labels):

        # make sure out we have a matching number of reviews and labels
        assert(len(training_reviews_raw) == len(training_labels))

        # Keep track of correct predictions to display accuracy during training
        correct_so_far = 0

        # Remember when we started for printing time statistics
        start = time.time()

        # preprocess training reviews
        training_reviews = [set([self.word2index[word] for word in review_raw.split(
            " ") if word in self.word2index.keys()]) for review_raw in training_reviews_raw]

        # loop through all the given reviews and run a forward and backward pass,
        # updating weights for every item
        for i in range(len(training_reviews)):

            # TODO: Get the next review and its correct label
            x = training_reviews[i]
            y = training_labels[i]

            # print(x)
            # print(y)

            # TODO: Implement the forward pass through the network.
            #       That means use the given review to update the input layer,
            #       then calculate values for the hidden layer,
            #       and finally calculate the output layer.
            #
            #       Do not use an activation function for the hidden layer,
            #       but use the sigmoid activation function for the output layer.

            # n_i = 72810
            # n_h = 10
            # n_o = 1

            self.layer_1 *= 0

            # hidden layer 1
            # weights_0_1.shape = n_i x n_h
            for index in x:
                self.layer_1 += self.weights_0_1[index]
            # layer_1.shape = (1 x n_i) x (n_i x n_h) = 1 x n_h
            # print("layer_1", layer_1.shape)

            # output layer 2
            # weights_1_2.shape = n_h x n_o
            layer_2 = self.sigmoid(np.dot(self.layer_1, self.weights_1_2))
            # layer_2.shape = (1 x n_h) x (n_h x n_o) = 1 x n_o
            # apply sigmoid -> value:[0,1]
            # print("layer_2", layer_2.shape)

            # TODO: Implement the back propagation pass here.
            #       That means calculate the error for the forward pass's prediction
            #       and update the weights in the network according to their
            #       contributions toward the error, as calculated via the
            #       gradient descent and back propagation algorithms you
            #       learned in class.

            # error = y - output
            # error.shape = 1 x 1
            error = self.get_target_for_label(y) - layer_2
            # print("error", error.shape)

            # error term for output
            # e * f'(x) = e * output * (1 - output)
            output_error_term = error * \
                self.sigmoid_output_2_derivative(layer_2)
            # output_error_term.shape = 1 x 1
            # print("output_error_term", output_error_term.shape)

            # propagate errors to hidden layer
            # hidden layer's contribution to the error
            # weights_1_2.shape = n_h x n_o
            # weights_1_2.T.shape = n_o x n_h
            hidden_error = np.dot(output_error_term, self.weights_1_2.T)
            # hidden_error.shape = n_o x n_h
            # print("hidden_error", hidden_error.shape)

            # error term for the hidden layer
            hidden_error_term = hidden_error
            # hidden_error_term.shape = n_o x n_h
            # print("hidden_error_term", hidden_error_term.shape)

            # Update weights
            # weights_1_2.shape = n_h x n_o
            # output_error_term.shape = 1 x 1
            # layer_1.shape = 1 x n_h
            self.weights_1_2 += self.learning_rate * \
                np.dot(self.layer_1.T, output_error_term)
            # weights_0_1.shape = n_i x n_h
            # hidden_error_term.shape = n_o x n_h

            for index in x:
                self.weights_0_1[index] += self.learning_rate * \
                    hidden_error_term[0]

            # TODO: Keep track of correct predictions. To determine if the prediction was
            #       correct, check that the absolute value of the output error
            #       is less than 0.5. If so, add one to the correct_so_far count.
            if np.abs(error) <= 0.5:
                correct_so_far += 1

            # For debug purposes, print out our prediction accuracy and speed
            # throughout the training process.

            elapsed_time = float(time.time() - start)
            reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0

            sys.stdout.write("\rProgress:" + str(100 * i/float(len(training_reviews)))[:4]
                             + "% Speed(reviews/sec):" +
                             str(reviews_per_second)[0:5]
                             + " #Correct:" +
                             str(correct_so_far) + " #Trained:" + str(i+1)
                             + " Training Accuracy:" + str(correct_so_far * 100 / float(i+1))[:4] + "%")
            if(i % 2500 == 0):
                print("")

    def test(self, testing_reviews, testing_labels):
        """
        Attempts to predict the labels for the given testing_reviews,
        and uses the test_labels to calculate the accuracy of those predictions.
        """

        # keep track of how many correct predictions we make
        correct = 0

        # we'll time how many predictions per second we make
        start = time.time()

        # Loop through each of the given reviews and call run to predict
        # its label.
        for i in range(len(testing_reviews)):
            pred = self.run(testing_reviews[i])
            if(pred == testing_labels[i]):
                correct += 1

            # For debug purposes, print out our prediction accuracy and speed
            # throughout the prediction process.

            elapsed_time = float(time.time() - start)
            reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0

            sys.stdout.write("\rProgress:" + str(100 * i/float(len(testing_reviews)))[:4]
                             + "% Speed(reviews/sec):" +
                             str(reviews_per_second)[0:5]
                             + " #Correct:" +
                             str(correct) + " #Tested:" + str(i+1)
                             + " Testing Accuracy:" + str(correct * 100 / float(i+1))[:4] + "%")

    def run(self, review):
        """
        Returns a POSITIVE or NEGATIVE prediction for the given review.
        """
        # TODO: Run a forward pass through the network, like you did in the
        #       "train" function. That means use the given review to
        #       update the input layer, then calculate values for the hidden layer,
        #       and finally calculate the output layer.
        #
        #       Note: The review passed into this function for prediction
        #             might come from anywhere, so you should convert it
        #             to lower case prior to using it.

        # input layer 0
        x = set([self.word2index[word]
                 for word in review.split(" ") if word in self.word2index.keys()])

        self.layer_1 *= 0
        for index in x:
            self.layer_1 += self.weights_0_1[index]

        # output layer 2
        layer_2 = self.sigmoid(np.dot(self.layer_1, self.weights_1_2))

        # TODO: The output layer should now contain a prediction.
        #       Return `POSITIVE` for predictions greater-than-or-equal-to `0.5`,
        #       and `NEGATIVE` otherwise.
        return 'POSITIVE' if layer_2[0] >= 0.5 else "NEGATIVE"
