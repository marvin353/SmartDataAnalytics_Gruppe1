import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pydot
from sklearn.metrics import confusion_matrix
pd.set_option('display.max_columns', 500)
#import keras_metrics as km
from numpy import loadtxt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import balanced_accuracy_score
import datetime
from keras.models import Sequential
import os


class Evaluator:


    def __init__(self):
        print("Evaluator started")
        self.safeModel = False
        self.modelDir = ""

    def predict(self, model, testX, testy):
        self.safeModel = False
        self.testX = testX
        self.testy = testy

        # predict probabilities
        self.probs = model.predict_proba(testX)

        # keep probabilities for the positive outcome only
        self.probs = self.probs[:, 0]
        # Define theshold as 0.5 (Result would be probability value between 0 and 1, but labels are only 0 and 1, if value less than 0.5 -> label 0, otherwise label 1)
        self.y_pred = model.predict(testX) > 0.5
        self.y_pred = self.y_pred.flatten()
        #print(self.y_pred)


    def predictGen(self, model, generator, testy, totalLength, batchsize):
        self.safeModel = False
        self.testy = testy

        # predict probabilities
        self.y_pred = model.predict_generator(generator, steps=np.ceil(totalLength / batchsize), verbose=1) > 0.0
        self.y_pred = self.y_pred.flatten()


    def evaluateROC(self):
        # calculate roc curve
        fpr, tpr, thresholds = roc_curve(self.testy, self.probs)
        yval = np.interp(0.001, fpr,tpr)
        print("YVal: " + str(yval))
        # plot no skill
        plt.plot([0, 1], [0, 1], linestyle='--')
        # plot the roc curve for the model
        plt.plot(fpr, tpr, marker='.')
        # show the plot
        if self.safeModel:
           plt.savefig(self.modelDir + "ROC.eps")

        plt.show()


    def evaluateAUC(self):
        auc = roc_auc_score(self.testy, self.probs)
        print('AUC: %.3f' % auc)

    def evaluateBAC(self):
        bac = balanced_accuracy_score(self.testy, self.y_pred)
        print('BAC: %.3f' % bac)
        return bac

    def evaluateCM(self):
        CM = confusion_matrix(self.testy, self.y_pred)
        CM_norm = CM.astype('float') / CM.sum(axis=1)[:, np.newaxis]

        plt.rcParams.update({'font.size': 16})
        fig = plt.figure()

        ax = plt.subplot()
        sns.heatmap(CM_norm, annot=True, ax=ax, cmap="Blues");  # annot=True to annotate cells

        # labels, title and ticks
        ax.set_xlabel('Predicted labels');
        ax.set_ylabel('True labels');
        ax.xaxis.set_ticklabels(['Failure', 'No Failure']);
        ax.yaxis.set_ticklabels(['Failure', 'No Failure']);

        if self.safeModel:
           plt.savefig(self.modelDir + "CM.png")

        plt.show()

    def evaluateACC(self, history):
        # summarize history for accuracy
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='lower right')
        plt.show()

    def evaluateLOSS(self, history):
        # summarize history for loss
        plt.plot(history.history['loss'])
        # plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper right')
        plt.show()

    def perf_measure(self,y_actual, y_hat):
        TP = 0.000000001
        FP = 0.000000001
        TN = 0.000000001
        FN = 0.000000001

        for i in range(len(y_hat)):
            if y_actual[i] == y_hat[i] == 1:
                TP += 1
            if y_hat[i] == 1 and y_actual[i] != y_hat[i]:
                FP += 1
            if y_actual[i] == y_hat[i] == 0:
                TN += 1
            if y_hat[i] == 0 and y_actual[i] != y_hat[i]:
                FN += 1

        return (TP, FP, TN, FN)

    def others(self):
        TP, FP, TN, FN = self.perf_measure(self.testy, self.y_pred)
        """FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
        FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
        TP = np.diag(confusion_matrix)
        TN = confusion_matrix.values.sum() - (FP + FN + TP)"""

        print("FP: " + str(FP))
        print("FN: " + str(FN))
        print("TP: " + str(TP))
        print("TN: " + str(TN))

        # Sensitivity, hit rate, recall, or true positive rate
        TPR = TP / (TP + FN)
        # Specificity or true negative rate
        TNR = TN / (TN + FP)
        # Precision or positive predictive value
        PPV = TP / (TP + FP)
        # Negative predictive value
        NPV = TN / (TN + FN)
        # Fall out or false positive rate or false accept rate
        FAR = FP / (FP + TN)
        # False negative rate or false reject rate
        FRR = FN / (TP + FN)
        # False discovery rate
        FDR = FP / (TP + FP)

        # Overall accuracy
        ACC = (TP + TN) / (TP + FP + FN + TN)
        BAC = (1/2) * ((TP / (TP + FN)) + (TN / (TN + FP)))

        print("TPR: " + str(TPR))
        print("TNR: " + str(TNR))
        print("PPV: " + str(PPV))
        print("NPV: " + str(NPV))
        print("FAR: " + str(FAR))
        print("FRR: " + str(FRR))
        print("FDR: " + str(FDR))

        if self.saveModel:
            file = open(self.modelDir + "Metrics.txt", "w")
            file.write("ACC: " + str(ACC) + "\n")
            file.write("BAC: " + str(BAC) + "\n")
            file.write("TPR: " + str(TPR) + "\n")
            file.write("TNR: " + str(TNR) + "\n")
            file.write("PPV: " + str(PPV) + "\n")
            file.write("NPV: " + str(NPV) + "\n")
            file.write("FAR: " + str(FAR) + "\n")
            file.write("FRR: " + str(FRR) + "\n")
            file.write("FDR: " + str(FDR) + "\n")
            file.close()


    def saveModel(self, model, name):

        self.safeModel = True
        path = "models/" + str(datetime.datetime.now().strftime('%d-%m-%Y_%H:%M:%S'))
        self.modelDir = path + "/"
        modelName = name + "model.h5"

        try:
            os.mkdir(path)
        except OSError:
            print("Creation of the directory %s failed" % path)
        else:
            print("Successfully created the directory %s " % path)

        # Save model
        print("Save Model: " + name + ", at: " + self.modelDir)
        model.save_weights(self.modelDir + modelName)

        # Save sodel summary
        summary = model.summary()
        print(summary)
        #file = open(self.modelDir + "summary_" + name + "model.txt", "w")
        #file.write(str(summary))
        #file.close()

        # Open the file
        with open(self.modelDir + "summary_" + name + "model.txt", 'w') as fh:
            # Pass the file handle in as a lambda function to make it callable
            model.summary(print_fn=lambda x: fh.write(x + '\n'))