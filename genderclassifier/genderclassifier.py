import numpy as np
import pandas

from keras.layers import Dense
from keras.models import load_model, Sequential
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix

class GenderClassifier(object):
    def __init__(self):
        self.__model = None
        self.__char_indexes = None
        self.__indexes_char = None
        self.__trigrams_indexes = None
        self.__indexes_trigrams = None
        self.__classes_indexes = None
        self.__indexes_classes = None
        self.__chars = None
        self.__trigrams = None
        self.__classes = None

    def load(self, model_filename):
        self.__model = load_model("%s.model" % model_filename)
        self.__chars = np.load("%s.cvocab.npy" % model_filename).tolist()
        self.__trigrams = np.load("%s.tvocab.npy" % model_filename).tolist()
        self.__classes = np.load("%s.classes.npy" % model_filename).tolist()

        self.__char_indexes = dict((c, i) for i, c in enumerate(self.__chars))
        self.__indexes_char = dict((i, c) for i, c in enumerate(self.__chars))

        self.__trigrams_indexes = dict((t, i) for i, t in enumerate(self.__trigrams))
        self.__indices_trigrams = dict((i, t) for i, t in enumerate(self.__trigrams))

        self.__classes_indexes = dict((c, i) for i, c in enumerate(self.__classes))
        self.__indexes_classes = dict((i, c) for i, c in enumerate(self.__classes))

    def save(self, model_filename):
        self.__model.save("%s.model" % model_filename)
        np.save("%s.tvocab" % model_filename, np.asarray(self.__trigrams))
        np.save("%s.cvocab" % model_filename, np.asarray(self.__chars))
        np.save("%s.classes" % model_filename, np.asarray(self.__classes))


    def __parse_name(self, name):
        x = np.zeros(len(self.__trigrams_indexes)+len(self.__char_indexes))
        if name in self.__trigrams_indexes.keys():
            x[self.__trigrams_indexes[name[len(name)-3:]]] == 1
        for c in set(name):
            if c in self.__char_indexes.keys():
                x[len(self.__trigrams_indexes)+self.__char_indexes[c]] = name.count(c)/float(len(name))
        return x


    def __load_dataset(self, dataset, train_split):
        names = dataset[:, 0]
        genders = dataset[:, 1]

        text = ''.join(names)
        self.__chars = sorted(list(set(text)))
        self.__char_indexes = dict((c, i) for i, c in enumerate(self.__chars))
        self.__indexes_char = dict((i, c) for i, c in enumerate(self.__chars))

        self.__trigrams = sorted(list(set([name[len(name)-3:] for name in names])))
        self.__trigrams_indexes = dict((t, i) for i, t in enumerate(self.__trigrams))
        self.__indices_trigrams = dict((i, t) for i, t in enumerate(self.__trigrams))

        self.__classes = sorted(list(set(genders)))
        self.__classes_indexes = dict((c, i) for i, c in enumerate(self.__classes))
        self.__indexes_classes = dict((i, c) for i, c in enumerate(self.__classes))

        train_size = int(len(dataset)*train_split)
        test_size = len(dataset) - train_size

        np.random.shuffle(dataset) # a little bit of magic
        
        train_data = []
        test_data = []
        gender_bool = False
        for name, gender in dataset.tolist():
            x = self.__parse_name(name)
            if len(test_data) < test_size:
                if (gender_bool and gender == 'F') or ((not gender_bool) and gender == 'M'): # 50-50 in validation set
                    test_data.append([x, self.__classes_indexes[gender]])
                    gender_bool = not gender_bool
                else:
                    train_data.append([x, self.__classes_indexes[gender]])
            else:
                train_data.append([x, self.__classes_indexes[gender]])
        self.__train_data = np.array(train_data)
        self.__test_data = np.array(test_data)

    def train(self, dataset, train_split=0.8, dense_size=32, learning_rate=0.001, batch_size=32, epochs=50, activation='relu'):
        self.__load_dataset(dataset, train_split)

        train_x = np.array(self.__train_data[:, 0].tolist())
        train_y = to_categorical(self.__train_data[:, 1], 2)

        test_x = np.array(self.__test_data[:, 0].tolist())
        test_y = to_categorical(self.__test_data[:, 1], 2)

        print(train_x.shape)
        self.__model = Sequential()
        self.__model.add(Dense(dense_size, input_dim=train_x.shape[1], activation=activation, init='glorot_uniform'))
        self.__model.add(Dense(train_y.shape[1], activation='softmax', init='glorot_uniform'))
        self.__model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['categorical_accuracy'])

        self.__model.fit(train_x, train_y, batch_size=batch_size, nb_epoch=epochs, validation_data=(test_x, test_y), verbose=2)

    def predict(self, data):
        predictions = []
        for name in data:
            x = np.array([self.__parse_name(name)])
            prediction = self.__model.predict([x])
            predictions.append(self.__indexes_classes[np.argmax(prediction)])
        return predictions
    
    def evaluate(self, dataset):
        predictions = self.predict(dataset[:,0])
        confusion_matrix = sklearn_confusion_matrix(dataset[:,1], predictions, labels=self.__classes)

        precisions = []
        recalls = []
        accuracies = []

        for gender in self.__classes:
            idx = self.__classes_indexes[gender]
            precision = 1
            recall = 1
            if np.sum(confusion_matrix[idx,:]) > 0:
                precision = confusion_matrix[idx][idx]/np.sum(confusion_matrix[idx,:])
            if np.sum(confusion_matrix[:, idx]) > 0:
                recall = confusion_matrix[idx][idx]/np.sum(confusion_matrix[:, idx])
            precisions.append(precision)
            recalls.append(recall)

        precision = np.mean(precisions)
        recall = np.mean(recalls)
        f1 = (2*(precision*recall))/float(precision+recall)
        accuracy = np.sum(confusion_matrix.diagonal())/float(np.sum(confusion_matrix))

        return precision, recall, accuracy, f1