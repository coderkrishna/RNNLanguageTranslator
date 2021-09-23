################################################################################
import numpy as np
import pandas as pd
import re
import string
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
import heapq
import sys

stopwords = ["के", "का", "एक", "में", "की", "है", "यह", "और", "से", "हैं", "को", "पर", "इस", "होता", "कि", "जो", "कर",
             "मे", "गया", "करने", "किया", "लिये", "अपने", "ने", "बनी", "नहीं", "तो", "ही", "या", "एवं", "दिया", "हो",
             "इसका", "था", "द्वारा", "हुआ", "तक", "साथ", "करना", "वाले", "बाद", "लिए", "आप", "कुछ", "सकते", "किसी",
             "ये", "इसके", "सबसे", "इसमें", "थे", "दो", "होने", "वह", "वे", "करते", "बहुत", "कहा", "वर्ग", "कई", "करें",
             "होती", "अपनी", "उनके", "थी", "यदि", "हुई", "जा", "ना", "इसे", "कहते", "जब", "होते", "कोई", "हुए", "व",
             "न", "अभी", "जैसे", "सभी", "करता", "उनकी", "तरह", "उस", "आदि", "कुल", "एस", "रहा", "इसकी", "सकता", "रहे",
             "उनका", "इसी", "रखें", "अपना", "पे", "उसके"]


class DataPreprocessor:
    def __init__(self):
        print("init")

    def preProcess(self, df):

        # remove duplicates
        df.drop_duplicates(subset=None, keep='first');

        df = shuffle(df)
        # convert to lowecase
        df['ENG'] = df['ENG'].apply(lambda x: x.lower())
        df['HINDI'] = df['HINDI'].apply(lambda x: x.lower())

        # remove quotes
        df['ENG'] = df['ENG'].apply(lambda x: re.sub("'", '', x))
        df['HINDI'] = df['HINDI'].apply(lambda x: re.sub("'", '', x))

        # remove punctuation
        exclude = set(string.punctuation)
        df['ENG'] = df['ENG'].apply(lambda x: ''.join(ch for ch in x if ch not in exclude))
        df['HINDI'] = df['HINDI'].apply(lambda x: ''.join(ch for ch in x if ch not in exclude))

        # Remove numbers
        remove_digits = str.maketrans('', '', string.digits)
        df['ENG'] = df['ENG'].apply(lambda x: x.translate(remove_digits))
        df['HINDI'] = df['HINDI'].apply(lambda x: x.translate(remove_digits))
        df['HINDI'] = df['HINDI'].apply(lambda x: re.sub("[०१२३४५६७८९।]", "", x))

        # remove spaces
        df['ENG'] = df['ENG'].apply(lambda x: x.strip())
        df['HINDI'] = df['HINDI'].apply(lambda x: x.strip())

        df['ENG'] = df['ENG'].apply(lambda x: re.sub(" +", " ", x))
        df['HINDI'] = df['HINDI'].apply(lambda x: re.sub(" +", " ", x))

        # add START_ and _END to the hindi sequence
        df['HINDI'] = df['HINDI'].apply(lambda x: 'START_ ' + x + ' _END')

        # Get English and Hindi unique words
        max_len_eng = 0
        max_len_hindi = 0
        all_end_words = set()
        for eng in df['ENG']:
            sen_word_list = eng.split()
            if max_len_eng < len(sen_word_list):
                max_len_eng = len(sen_word_list)
            for word in sen_word_list:
                if word not in all_end_words:
                    all_end_words.add(word)

        self.max_len_eng = max_len_eng

        all_hindi_words = set()
        for hindi in df['HINDI']:
            sen_word_list = hindi.split()
            if max_len_hindi < len(sen_word_list):
                max_len_hindi = len(sen_word_list)
            for word in hindi.split():
                if word not in all_hindi_words:
                    all_hindi_words.add(word)
        self.max_len_hindi = max_len_hindi


        # converting into english and hindi words list
        input_words = sorted(list(all_end_words))
        output_words = sorted(list(all_hindi_words))
        self.input_words = input_words
        self.output_words = output_words


        # integer encode
        label_encoder = LabelEncoder()
        input_token_index = label_encoder.fit_transform(input_words)
        target_token_index = label_encoder.fit_transform(output_words)

        # binary encode
        onehot_encoder = OneHotEncoder(sparse=False)
        input_token_index = input_token_index.reshape(len(input_token_index), 1)
        self.input_onehot_encoded = onehot_encoder.fit_transform(input_token_index)
        target_token_index = target_token_index.reshape(len(target_token_index), 1)
        self.output_onehot_encoded = onehot_encoder.fit_transform(target_token_index)

        if (len(input_words) < len(output_words)):
            diff = len(output_words) - len(input_words)
            padding_zeros = np.zeros((1, len(input_words)))
            for i in range(diff):
                self.input_onehot_encoded = np.concatenate((self.input_onehot_encoded, padding_zeros.T), axis=1)
        else:
            diff = len(input_words) - len(output_words)
            padding_zeros = np.zeros((1, len(output_words)))
            for i in range(diff):
                self.output_onehot_encoded = np.concatenate((self.output_onehot_encoded, padding_zeros.T), axis=1)

        # convert dataframe to ndarray
        dataEng = df['ENG'].to_numpy()
        dataHindi = df['HINDI'].to_numpy()

        # split dataset
        indices = np.random.permutation(dataEng.shape[0])

        training_indices, test_indices = indices[:int((dataEng.shape[0] * 0.8))], indices[
                                                                                  int((dataEng.shape[0] * 0.8)):]
        X_train, X_test = dataEng[training_indices], dataEng[test_indices]
        y_train, y_test = dataHindi[training_indices], dataHindi[test_indices]

        return X_train, X_test, y_train, y_test, max_len_eng, max_len_hindi, len(all_end_words), len(all_hindi_words)


class RecurrentNeuralNet:
    def __init__(self, max_eng_len, max_hin_len, eng_unique_words_count, hin_unique_words_count, input_words,
                 input_onehot_encoded, output_words, output_onehot_encoded):
        self.max_eng_len = max_eng_len
        self.max_hin_len = max_hin_len

        self.input_words = input_words
        self.input_onehot_encoded = input_onehot_encoded
        self.output_words = output_words
        self.output_onehot_encoded = output_onehot_encoded

        self.eng_unique_words_count = eng_unique_words_count
        self.hin_unique_words_count = hin_unique_words_count

        np.random.seed(1)
        max_len = 0
        if self.eng_unique_words_count < self.hin_unique_words_count:
            max_len = self.hin_unique_words_count
        else:
            max_len = self.eng_unique_words_count

        self.max_len = max_len

        # weights and bias
        self.Wa = np.random.rand(max_len, 1)
        self.Wi = np.random.rand(max_len, 1)
        # self.EWf = np.random.rand(eng_unique_words_count)
        self.Wo = np.random.rand(max_len, 1)
        self.Wreca = np.random.rand(1)
        self.Wreci = np.random.rand(1)
        # self.EWrecf = np.random.rand(1)
        self.Wreco = np.random.rand(1)
        self.ba = np.random.rand(1)
        self.bi = np.random.rand(1)
        # self.Ebf = np.random.rand(1)
        self.bo = np.random.rand(1)
        self.DV = np.random.rand(max_len, 1)
        print("**** Initialized network parameters ****")

        # weights and bias

    def activation_sigmoid(self, x):
        """
        This function returns the values after applying sigmoid function
        :param x: input value
        :return: sigmoid(x)
        """
        sig = 1 / (1 + np.exp(-x, dtype=np.float64))
        sig = np.minimum(sig, 0.9999)  # Set upper bound
        sig = np.maximum(sig, 0.0001)  # Set lower bound
        return sig

    def activation_tanh(self, x):
        """
        This function returns the value after applying tanh
        :param x: input value
        :return: tanh(x)
        """
        tanh_value = (2 * self.activation_sigmoid(2 * x)) - 1
        return tanh_value

    def calculate_derivative_of_tanh(self, net):
        """
        This function returns the value after applying (1 - tanh(net)square)
        :param net: 
        :return: 
        """
        delta_output = (1 - np.square(np.tanh(net)))
        return delta_output

    def softmax(self, x):
        """
        This function returns the values after applying softmax function
        :param x: input value
        :return: softmax(x)
        """

        output_values = self.DV * x
        output_values[0] = 0
        output_values[1] = 0
        e_x = np.exp(output_values - np.max(output_values))
        return np.transpose(e_x/e_x.sum())

    def cross_entropy(self, predictions, targets, epsilon=1e-12):
        """
        This function returns the cross entropy value for the given predictions and targets
        :param predictions: predicted output
        :param targets: expected output
        :param epsilon: value of epsilon
        :return: cross entropy
        """
        predictions = np.clip(predictions, epsilon, 1. - epsilon)
        N = predictions.shape[0]
        j = self.findPosition(targets)
        ce = -np.sum(targets[j] * np.log(predictions[j] + 1e-9)) / N
        return ce

    def total_cross_entropy(self, predictions, targets):
        """
        Computes total cross entropy for entire sentence
        :param predictions: array of predictions for each word in sentence
        :param targets: array of target values for each word in sentence
        :return: total cross entropy
        """
        error = 0;
        T = self.max_hin_len
        for i in range(T - 1):
            """teachers forcing technique """
            error = error + self.cross_entropy(predictions[i], targets[i + 1])
        return error

    def findPosition(self, array):
        """
        Method returns index of element having having value 1 in tagret one-hot vector 
        :param array: input one hot vector
        :return: 
        """
        l = len(array)
        for i in range(l):
            if array[i] == 1.0:
                return i
        return 0;

    def embeddingLayer(self, X, RNNType):
        """
        Method returns array of one hot encoded vectors for input X
        :param X: Input sentence
        :param RNNType: Specifies whether it is encoder(for English) or decoder(for Hindi)
        :return: Array of one hot encoded representations for input sentence X
        """
        # print(X)
        encoded_array = []
        count = 0;
        if RNNType == "encoder":
            en_len = self.input_onehot_encoded.shape[1]
            padding_zeros = [0] * en_len
            words_in_sentence = X.split()
            for word in words_in_sentence:
                encoded_array.append(self.input_onehot_encoded[self.input_words.index(word)])
                count = count + 1

            while count < self.max_eng_len:
                encoded_array.append(padding_zeros)
                count = count + 1
        else:
            hindi_len = self.output_onehot_encoded.shape[1]
            padding_zeros = [0] * hindi_len
            words_in_sentence = X.split()
            for word in words_in_sentence:
                encoded_array.append(self.output_onehot_encoded[self.output_words.index(word)])
                count = count + 1

            while count < self.max_hin_len:
                encoded_array.append(padding_zeros)
                count = count + 1

        return np.array(encoded_array)


    def forward_pass(self, X, RNNType, cellInternalState0, outputHiddenState0):
        """
        Method performs the forward pass through RNN
        :param X: Input sentence
        :param RNNType: phase (encoder,decoder_training or decoder_testing)
        :param cellInternalState0: previous internal state
        :param outputHiddenState0: previous hidden state
        :return: cellinternal state, current hidden state and output value
        """
        # Encoder
        encoded = self.embeddingLayer(X, RNNType)
        # print(encoded)

        if RNNType == "encoder":
            T = self.max_eng_len
            cell_internal_s = np.zeros((T, 1))
            o_hiddenState = np.zeros((T, 1))
            o = np.zeros((T - 1, 1))

            for t in np.arange(T):
                a = self.activation_tanh(
                    np.dot(encoded[t], self.Wa) + np.dot(self.Wreca, o_hiddenState[t - 1]) + self.ba)
                i = self.activation_sigmoid(
                    np.dot(encoded[t], self.Wi) + np.dot(self.Wreci, o_hiddenState[t - 1]) + self.bi)
                oo = self.activation_sigmoid(
                    np.dot(encoded[t], self.Wo) + np.dot(self.Wreco, o_hiddenState[t - 1]) + self.bo)
                cell_internal_s[t] = a * i + cell_internal_s[t - 1]
                o_hiddenState[t] = self.activation_tanh(cell_internal_s[t]) * oo
        elif RNNType == "decoder_training":
            T = self.max_hin_len
            cell_internal_s = np.zeros((T, 1))  # T-1
            o_hiddenState = np.zeros((T, 1))  # T-1
            o = np.zeros((T, self.max_len))  # T-1
            hwords = X.split(" ")

            a = self.activation_tanh(np.dot(encoded[0], self.Wa) + np.dot(self.Wreca, outputHiddenState0) + self.ba)
            i = self.activation_sigmoid(np.dot(encoded[0], self.Wi) + np.dot(self.Wreci, outputHiddenState0) + self.bi)
            oo = self.activation_sigmoid(np.dot(encoded[0], self.Wo) + np.dot(self.Wreco, outputHiddenState0) + self.bo)
            # cell_internal_s[0] = a*i+f*cellInternalState0
            cell_internal_s[0] = a * i + cellInternalState0
            o_hiddenState[0] = self.activation_tanh(cell_internal_s[0]) * oo
            o[0] = self.softmax(o_hiddenState[0])

            for t in np.arange(1, T - 1):  # T-1
                a = self.activation_tanh(
                    np.dot(encoded[t], self.Wa) + np.dot(self.Wreca, o_hiddenState[t - 1]) + self.ba)
                i = self.activation_sigmoid(
                    np.dot(encoded[t], self.Wi) + np.dot(self.Wreci, o_hiddenState[t - 1]) + self.bi)
                oo = self.activation_sigmoid(
                    np.dot(encoded[t], self.Wo) + np.dot(self.Wreco, o_hiddenState[t - 1]) + self.bo)
                cell_internal_s[t] = a * i + cell_internal_s[t - 1]
                o_hiddenState[t] = self.activation_tanh(cell_internal_s[t]) * oo
                o[t] = self.softmax(o_hiddenState[t])
                if np.any(encoded[t + 1]) and (hwords[t + 1] not in stopwords):
                    # run backward pass for every time step t
                    self.backward_pass_exp(a, i, oo, cell_internal_s[t], o_hiddenState[t], encoded[t], o[t],
                                           encoded[t + 1])
        elif RNNType == "decoder_testing":
            # The total number of time steps
            T = self.max_hin_len
            # To save hidden states
            cell_internal_s = np.zeros((T, 1))  # T-1
            # To save output hidden states
            o_hiddenState = np.zeros((T, 1))  # T-1
            # To save the output state
            o = np.zeros((T, self.max_len))  # T-1
            o[0][0] = 1 # set it to START_
            a = self.activation_tanh(np.dot(o[0], self.Wa) + np.dot(self.Wreca, outputHiddenState0) + self.ba)
            i = self.activation_sigmoid(np.dot(o[0], self.Wi) + np.dot(self.Wreci, outputHiddenState0) + self.bi)
            oo = self.activation_sigmoid(np.dot(o[0], self.Wo) + np.dot(self.Wreco, outputHiddenState0) + self.bo)
            cell_internal_s[0] = a * i + cellInternalState0
            # o_hiddenState[0] = self.activation_tanh(cell_internal_s[0]) * oo
            o_hiddenState[0] = self.activation_tanh(cell_internal_s[0]) * oo
            o[0] = self.softmax(o_hiddenState[0])

            for t in np.arange(1, T - 1):  # T-1
                a = self.activation_tanh(
                    np.dot(o[t-1], self.Wa) + np.dot(self.Wreca, o_hiddenState[t - 1]) + self.ba)
                i = self.activation_sigmoid(
                    np.dot(o[t-1], self.Wi) + np.dot(self.Wreci, o_hiddenState[t - 1]) + self.bi)
                oo = self.activation_sigmoid(
                    np.dot(o[t-1], self.Wo) + np.dot(self.Wreco, o_hiddenState[t - 1]) + self.bo)
                cell_internal_s[t] = a * i + cell_internal_s[t - 1]
                o_hiddenState[t] = self.activation_tanh(cell_internal_s[t]) * oo
                o[t] = self.softmax(o_hiddenState[t])

        return cell_internal_s, o_hiddenState, o

    def backward_pass_exp(self, a, i, oo, cell_internal_state, hidden_state, currentX, currentY, expectedY,
                          learning_rate=0.01):
        # update softmax weight
        # calculate delta for softmax weights
        # print("---------- running backward pass ---------")
        softMax_error_intermediate = currentY - expectedY
        softmax_delta = self.calculate_delta_for_softmax_weight(hidden_state, expectedY, currentY)
        # print("delta softmax:",softmax_delta)
        # -------------------- update for  Wa -------------------------
        # (1- tanhsq(P)) = 1 - sq(a)
        # (1 - tanhsq(cell_internal_state)) = calculate_derivative_of_tanh(cell_internal_state)
        cell_internal_state_derivative = self.calculate_derivative_of_tanh(cell_internal_state)
        # cell_internal_state_derivative = 1
        # if(cell_internal_state<=0):
        #     cell_internal_state_derivative = 0

        dhdWa_intermediate_value = (oo * cell_internal_state_derivative * i * (1 - np.square(a)))
        dhdWa = dhdWa_intermediate_value * currentX
        dEdWa = softMax_error_intermediate * dhdWa
        # print("delta Wa:",dEdWa)
        # ---------------------- update for Wa ends ---------------------
        dhdWreca = dhdWa_intermediate_value * hidden_state
        dEdWreca = np.sum(softMax_error_intermediate * dhdWreca)
        # print("delta Wreca:",dEdWreca)
        # ---------------------- update for Wreca ends ------------------
        # ---------------------- update Wi ------------------------------
        dhdWi_intermediate_value = oo * cell_internal_state_derivative * a * (i * (1 - i))
        dhdWi = dhdWi_intermediate_value * currentX
        dEdWi = softMax_error_intermediate * dhdWi
        # print("delta Wi:",dEdWi)
        # --------------------- update for Wi ends ----------------------
        dhdWreci = dhdWi_intermediate_value * hidden_state
        dEdWreci = np.sum(softMax_error_intermediate * dhdWreci)
        # print("delta Wreci:",dEdWreci)
        # --------------------- update for Wreci ends -------------------
        # --------------------- update Wo -------------------------------
        dhdWoo_intermedate_value = np.tanh(cell_internal_state) * oo * (1 - oo)
        dhdWoo = dhdWoo_intermedate_value * currentX
        dEdWoo = softMax_error_intermediate * dhdWoo
        # print("delta Wo:",dEdWoo)
        # --------------------- update for Wo ends ---------------------
        dhdWreco = dhdWoo_intermedate_value * hidden_state
        dEdWreco = np.sum(softMax_error_intermediate * dhdWreco)
        # print("delta Wreco:",dEdWreco)
        # print("---------- calculated delta values  ---------")
        # print("---------- updating weights  ---------")
        # print("Wa",self.Wa)
        # print("deltaW", dEdWa)
        self.Wa = self.Wa - (learning_rate * dEdWa).reshape(self.max_len, 1)
        self.Wreca = self.Wreca - (learning_rate * dEdWreca)
        self.Wi = self.Wi - (learning_rate * dEdWi).reshape(self.max_len, 1)
        self.Wreci = self.Wreci - (learning_rate * dEdWreci)
        self.Wo = self.Wo - (learning_rate * dEdWoo).reshape(self.max_len, 1)
        self.Wreco = self.Wreco - (learning_rate * dEdWreco)
        self.DV = self.DV - (learning_rate * softmax_delta).reshape(self.max_len, 1)
        # print("---------- weights updated successfully  ---------")


    def calculate_delta_for_softmax_weight(self, h, t, y):
        """
        Method to calculate derivative for output softmax layer
        :param h: previous hidden state
        :param t: target output
        :param y: predicted output
        :return: dEdV derivative of error with respect to output softmax Weights
        """
        dEdV = np.zeros(self.DV.shape)
        dEdV = ((y - t) * h)
        # print(dEdV)
        return dEdV

    def trainNetwork(self, X, Y, epoch, X_test, y_test):
        """
        Method to train the network
        :param X: Input training data set
        :param Y: Input training expected outputs
        :param epoch: number of epochs
        :param X_test: Test data
        :param y_test: Expected outputs for test data
        :return: array of values of train error and test error for every epoch
        """
        trainerrorarray = []
        testerrorarray = []
        for count in range(epoch):
            print(" ---------------- running epoch :", count)
            sum_of_errors = 0
            for datapoint in range(X.size):
                itercount = 1
                if (len(X[datapoint].split()) < 3):
                    # to balance the length of sentences we had increased count of smaller sentences
                    # by repeating each small sentence by 10
                    itercount = 10
                for iter in range(itercount):
                    E_cell_internal_state, E_output_hiddenState, E_output = self.forward_pass(X[datapoint], "encoder",
                                                                                              0, 0)
                    D_cell_internal_state, D_output_hiddenState, D_outputT = self.forward_pass(Y[datapoint],
                                                                                               "decoder_training",
                                                                                               E_cell_internal_state[
                                                                                                   E_cell_internal_state.shape[
                                                                                                       0] - 1],
                                                                                               E_output_hiddenState[
                                                                                                   E_output_hiddenState.shape[
                                                                                                       0] - 1])
                    # print(D_outputT)
                    Yencoded = self.embeddingLayer(Y[datapoint], "decoder")
                    totalError = self.total_cross_entropy(D_outputT, Yencoded)
                    sum_of_errors = sum_of_errors + totalError
                    # print(" ---------------- epoch :", count," ---- ends")
            print("average train error for epoch : ", sum_of_errors / X.size)
            trainerrorarray.append(sum_of_errors / X.size)
            testerrorarray.append(self.test_network(X_test, y_test, False))
        return trainerrorarray, testerrorarray

    def test_network(self, X_test, y_test, printsent):
        sum_of_errors = 0
        for datapoint in range(X_test.size):
            E_cell_internal_state, E_output_hiddenState, E_output = self.forward_pass(X_test[datapoint], "encoder", 0,
                                                                                      0)
            D_cell_internal_state, D_output_hiddenState, D_outputT = self.forward_pass(y_test[datapoint],
                                                                                       "decoder_testing",
                                                                                       E_cell_internal_state[
                                                                                           E_cell_internal_state.shape[
                                                                                               0] - 1],
                                                                                       E_output_hiddenState[
                                                                                           E_output_hiddenState.shape[
                                                                                               0] - 1])
            Yencoded = self.embeddingLayer(y_test[datapoint], "decoder")
            totalError = self.total_cross_entropy(D_outputT, Yencoded)
            sum_of_errors = sum_of_errors + totalError
            if printsent:
                print("Input Sentence : ", X_test[datapoint])
                print("Expected Output Sentence : ", y_test[datapoint])
                print("Predicted Sentence : ", self.convert_output_to_sentence(D_outputT))
                for encodedY in range(len(y_test[datapoint].split())) :
                    j = self.findPosition(Yencoded[encodedY])
                    print(" Expected Word :",self.output_words[j])
                    print(" Predicted Probability:",D_outputT[encodedY][j])


        print("average test error : ", sum_of_errors / X_test.size)
        return (sum_of_errors / X_test.size)

    def convert_output_to_sentence(self, softmax_output):
        T = self.max_hin_len
        output_sentence = ""
        for i in range(T - 1):
            max_probability_index = np.argmax(softmax_output[i])
            indices = heapq.nlargest(3, range(len(softmax_output[i])), softmax_output[i].__getitem__)
            # max occurence of _START_ and _END_ force network to lear those hence output of third most probable word.
            output_sentence = output_sentence + " " + self.output_words[max_probability_index]
        return output_sentence


    def getTranslation(self,inputsentence):
        E_cell_internal_state, E_output_hiddenState, E_output = self.forward_pass(inputsentence, "encoder", 0,
                                                                                  0)

        # The total number of time steps
        T = self.max_hin_len
        # To save hidden states
        cell_internal_s = np.zeros((T, 1))  # T-1
        # To save output hidden states
        o_hiddenState = np.zeros((T, 1))  # T-1
        # To save the output state
        o = np.zeros((T, self.max_len))  # T-1

        a = self.activation_tanh(np.dot(o[0], self.Wa) + np.dot(self.Wreca, E_output_hiddenState[E_output_hiddenState.shape[0] - 1]) + self.ba)
        i = self.activation_sigmoid(np.dot(o[0], self.Wi) + np.dot(self.Wreci, E_output_hiddenState[E_output_hiddenState.shape[0] - 1]) + self.bi)
        oo = self.activation_sigmoid(np.dot(o[0], self.Wo) + np.dot(self.Wreco, E_output_hiddenState[E_output_hiddenState.shape[0] - 1]) + self.bo)
        cell_internal_s[0] = a * i + E_cell_internal_state[E_cell_internal_state.shape[0] - 1]
        o_hiddenState[0] = self.activation_tanh(cell_internal_s[0]) * oo
        o[0] = self.softmax(o_hiddenState[0])

        for t in np.arange(1, T - 1):  # T-1
            a = self.activation_tanh(
                np.dot(o[t - 1], self.Wa) + np.dot(self.Wreca, o_hiddenState[t - 1]) + self.ba)
            i = self.activation_sigmoid(
                np.dot(o[t - 1], self.Wi) + np.dot(self.Wreci, o_hiddenState[t - 1]) + self.bi)
            oo = self.activation_sigmoid(
                np.dot(o[t - 1], self.Wo) + np.dot(self.Wreco, o_hiddenState[t - 1]) + self.bo)
            cell_internal_s[t] = a * i + cell_internal_s[t - 1]
            o_hiddenState[t] = self.activation_sigmoid(cell_internal_s[t]) * oo
            o[t] = self.softmax(o_hiddenState[t])
        sent = self.convert_output_to_sentence(o)
        print(sent)

if __name__ == "__main__":
    dataset_file_name = 'data/hin_combined.csv'
    #   Load data from csv file
    df = pd.read_csv(filepath_or_buffer=dataset_file_name, sep='\t', header=None)
    df.columns = ["ENG", "HINDI"]
    dp_ins = DataPreprocessor()
    X_train, X_test, y_train, y_test, max_eng_len, max_hindi_len, eng_unique_words_count, hin_unique_words_count = dp_ins.preProcess(
        df)

    rnn = RecurrentNeuralNet(max_eng_len, max_hindi_len, eng_unique_words_count, hin_unique_words_count,
                             dp_ins.input_words, dp_ins.input_onehot_encoded, dp_ins.output_words,
                             dp_ins.output_onehot_encoded)
    rnn.Wa = np.load("weights/Wa.npy").reshape(hin_unique_words_count, 1)
    rnn.Wreca = np.load("weights/Wreca.npy")
    rnn.Wo = np.load("weights/Wo.npy").reshape(hin_unique_words_count, 1)
    rnn.Wreco = np.load("weights/Wreco.npy")
    rnn.Wi = np.load("weights/Wi.npy").reshape(hin_unique_words_count, 1)
    rnn.Wreci = np.load("weights/Wreci.npy")
    rnn.DV = np.load("weights/DV.npy").reshape(hin_unique_words_count, 1)
    rnn.ba = np.load("weights/ba.npy")
    rnn.bi = np.load("weights/bi.npy")
    rnn.bo = np.load("weights/bo.npy")

    do_train = False
    if len(sys.argv) == 2:
        do_train = (sys.argv[1] == 'True')

    if do_train:
        trainerror, testerror = rnn.trainNetwork(X_train, y_train,10, X_test, y_test)
        rnn.test_network(X_test[0:10], y_test[0:10], True)
        print("Training error array", trainerror)
        print("Test error array", testerror)
        np.save("trainerror", trainerror)
        np.save("testerror", testerror)
        np.save("Wa", rnn.Wa)
        np.save("Wreca", rnn.Wreca)
        np.save("Wo", rnn.Wo)
        np.save("Wreco", rnn.Wreco)
        np.save("Wi", rnn.Wi)
        np.save("Wreci", rnn.Wreci)
        np.save("DV", rnn.DV)
        np.save("ba", rnn.ba)
        np.save("bi", rnn.bi)
        np.save("bo", rnn.bo)
    else:
        avgtesterrorvalue = rnn.test_network(X_test[0:5], y_test[0:5], True)

