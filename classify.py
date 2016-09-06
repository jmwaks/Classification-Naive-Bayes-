"""
Assignment 3. Implement a Multinomial Naive Bayes classifier for spam filtering.

You'll only have to implement 3 methods below:

train: compute the word probabilities and class priors given a list of documents labeled as spam or ham.
classify: compute the predicted class label for a list of documents
evaluate: compute the accuracy of the predicted class labels.

"""

from collections import defaultdict
from collections import Counter
import glob
import math
import os



class Document(object):
    """ A Document. Do not modify.
    The instance variables are:

    filename....The path of the file for this document.
    label.......The true class label ('spam' or 'ham'), determined by whether the filename contains the string 'spmsg'
    tokens......A list of token strings.
    """

    def __init__(self, filename=None, label=None, tokens=None):
        """ Initialize a document either from a file, in which case the label
        comes from the file name, or from specified label and tokens, but not
        both.
        """
        if label: # specify from label/tokens, for testing.
            self.label = label
            self.tokens = tokens
        else: # specify from file.
            self.filename = filename
            self.label = 'spam' if 'spmsg' in filename else 'ham'
            self.tokenize()

    def tokenize(self):
        self.tokens = ' '.join(open(self.filename).readlines()).split()


class NaiveBayes(object):

    def get_word_probability(self, label, term):
        """
        Return Pr(term|label). This is only valid after .train has been called.

        Params:
          label: class label.
          term: the term
        Returns:
          A float representing the probability of this term for the specified class.

        >>> docs = [Document(label='spam', tokens=['a', 'b']), Document(label='spam', tokens=['b', 'c']), Document(label='ham', tokens=['c', 'd'])]
        >>> nb = NaiveBayes()
        >>> nb.train(docs)
        >>> nb.get_word_probability('spam', 'a')
        0.25
        >>> nb.get_word_probability('spam', 'b')
        0.375
        """
        if label == 'spam':
            prb = self.condprob_spam[term]
        elif label == 'ham':
            prb = self.condprob_ham[term]
        return prb



    def get_top_words(self, label, n):
        """ Return the top n words for the specified class, using the odds ratio.
        The score for term t in class c is: p(t|c) / p(t|c'), where c'!=c.

        Params:
          labels...Class label.
          n........Number of values to return.
        Returns:
          A list of (float, string) tuples, where each float is the odds ratio
          defined above, and the string is the corresponding term.  This list
          should be sorted in descending order of odds ratio.

        >>> docs = [Document(label='spam', tokens=['a', 'b']), Document(label='spam', tokens=['b', 'c']), Document(label='ham', tokens=['c', 'd'])]
        >>> nb = NaiveBayes()
        >>> nb.train(docs)
        >>> nb.get_top_words('spam', 2)
        [(2.25, 'b'), (1.5, 'a')]
        """
        top_words = [()]
        if label == 'spam':
            for term in self.condprob_spam:
                odds = self.condprob_spam[term]/self.condprob_ham[term]
                top_words.append((odds,term))
        elif label == 'ham':
            for term in self.condprob_ham:
                odds = self.condprob_ham[term] / self.condprob_spam[term]
                top_words.append((odds, term))
        top_words = sorted(top_words, reverse = True)
        return top_words[0:n]


    def train(self, documents):
        """
        Given a list of labeled Document objects, compute the class priors and
        word conditional probabilities, following Figure 13.2 of your
        book. Store these as instance variables, to be used by the classify
        method subsequently.
        Params:
          documents...A list of training Documents.
        Returns:
          Nothing.

        doc_counts = Counter(documents)
        spam_counts = Counter()
        ham_counts = Counter()
        N = 0
        for docs in doc_counts:
            N = doc_counts[docs] + N
        label_d = docs.label
        tokenz = docs.tokens
        """
        spam_counts = Counter()
        ham_counts = Counter()
        N_cs = 0
        N_ch = 0
        N = 0
        self.V = set()
        text_cs = []
        text_ch = []
        self.condprob_ham = Counter()
        self.condprob_spam = Counter()
        for doc in documents:
            N += 1
            label_d = doc.label
            tokenz = doc.tokens
            if label_d == 'spam':
                for token in tokenz:
                    spam_counts[token] += 1
                    text_cs.append(token)
                    self.V.add(token)
                N_cs += 1
            elif label_d == 'ham':
                for token in tokenz:
                    ham_counts[token] += 1
                    text_ch.append(token)
                    self.V.add(token)
                N_ch += 1
        self.prior_ham = N_ch / N
        self.prior_spam = N_cs / N
        for term in self.V:
            Tct_ham = ham_counts[term]
            self.condprob_ham[term] = (Tct_ham + 1.0) / (len(text_ch) + len(self.V))
            Tct_spam = spam_counts[term]
            self.condprob_spam[term] = (Tct_spam + 1.0) / (len(text_cs) + len(self.V))


    def classify(self, documents):
        """ Return a list of strings, either 'spam' or 'ham', for each document.
        Params:
          documents....A list of Document objects to be classified.
        Returns:
          A list of label strings corresponding to the predictions for each document.
        """
        labels = []
        for doc in documents:
            tokenz = doc.tokens
            logscore_spam = math.log10(self.prior_spam)
            logsocre_ham = math.log10(self.prior_ham)
            for token in tokenz:
                if token in self.condprob_ham and self.condprob_spam:
                    logscore_spam += math.log10(self.condprob_spam[token])
                    logsocre_ham += math.log10(self.condprob_ham[token])
            if logscore_spam > logsocre_ham:
                labels.append('spam')
            else:
                labels.append('ham')
        return labels


def evaluate(predictions, documents):
    """ Evaluate the accuracy of a set of predictions.
    Return a tuple of three values (X, Y, Z) where
    X = percent of documents classified correctly
    Y = number of ham documents incorrectly classified as spam
    X = number of spam documents incorrectly classified as ham

    Params:
      predictions....list of document labels predicted by a classifier.
      documents......list of Document objects, with known labels.
    Returns:
      Tuple of three floats, defined above.
    """
    counter = -1
    correct = []
    incorrect = []
    Y = 0
    Z = 0
    for doc in documents:
        counter += 1
        if predictions[counter] == doc.label:
            correct.append(predictions[counter])
        else:
            incorrect.append(predictions[counter])
    for cor in incorrect:
        if cor == 'spam':
            Y += 1
        else:
            Z += 1
    X = (len(correct) / (len(correct) + len(incorrect)))
    return [X, Y, Z]



def main():
    """ Do not modify. """
    if not os.path.exists('train'):  # download data
       from urllib.request import urlretrieve
       import tarfile
       urlretrieve('http://cs.iit.edu/~culotta/cs429/lingspam.tgz', 'lingspam.tgz')
       tar = tarfile.open('lingspam.tgz')
       tar.extractall()
       tar.close()
    train_docs = [Document(filename=f) for f in glob.glob("train/*.txt")]
    print('read', len(train_docs), 'training documents.')
    nb = NaiveBayes()
    nb.train(train_docs)
    test_docs = [Document(filename=f) for f in glob.glob("test/*.txt")]
    print('read', len(test_docs), 'testing documents.')
    predictions = nb.classify(test_docs)
    results = evaluate(predictions, test_docs)
    print('accuracy=%.3f, %d false spam, %d missed spam' % (results[0], results[1], results[2]))
    print('top ham terms: %s' % ' '.join('%.2f/%s' % (v,t) for v, t in nb.get_top_words('ham', 10)))
    print('top spam terms: %s' % ' '.join('%.2f/%s' % (v,t) for v, t in nb.get_top_words('spam', 10)))

if __name__ == '__main__':
    main()
