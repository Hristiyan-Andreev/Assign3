import numpy as np
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix

from svm_plot import plot_svm_decision_boundary, plot_score_vs_degree, plot_score_vs_gamma, plot_mnist, \
    plot_confusion_matrix

"""
Computational Intelligence TU - Graz
Assignment 3: Support Vector Machine, Kernels & Multiclass classification
Part 1: SVM, Kernels

TODOS are all contained here.
"""

__author__ = 'bellec,subramoney'


def ex_1_a(x, y):
    """
    Solution for exercise 1 a)
    :param x: The x values
    :param y: The y values
    :return:
    """
    ###########
    ## TODO:
    ## Train an SVM with a linear kernel
    ## and plot the decision boundary and support vectors using 'plot_svm_decision_boundary' function
    SVM = svm.SVC(kernel='linear')
    SVM.fit(x, y)

    plot_svm_decision_boundary(SVM, x, y, x_test=None, y_test=None)


    ###########
    pass


def ex_1_b(x, y):
    """
    Solution for exercise 1 b)
    :param x: The x values
    :param y: The y values
    :return:
    """
    ###########
    ## TODO:
    ## Add a point (4,0) with label 1 to the data set and then
    ## train an SVM with a linear kernel
    ## and plot the decision boundary and support vectors using 'plot_svm_decision_boundary' function
    ###########
# Add (4,0)

    x = np.append(x,[[4,0]], axis=0)
    y = np.append(y,1)

    SVM = svm.SVC(kernel='linear')
    SVM.fit(x, y)

    plot_svm_decision_boundary(SVM, x, y, x_test=None, y_test=None)
    pass


def ex_1_c(x, y):
    """
    Solution for exercise 1 c)
    :param x: The x values
    :param y: The y values
    :return:
    """
    ###########
    ## TODO:
    ## Add a point (4,0) with label 1 to the data set and then
    ## train an SVM with a linear kernel with different values of C
    ## and plot the decision boundary and support vectors  for each using 'plot_svm_decision_boundary' function
    ###########
    x = np.append(x, [[4, 0]], axis=0)
    y = np.append(y, 1)

    Cs = [1e6, 1, 0.1, 0.001]
    for C in Cs:
        SVM = svm.SVC(C = C, kernel='linear')
        SVM.fit(x, y)
        plot_svm_decision_boundary(SVM, x, y, x_test=None, y_test=None)


def ex_2_a(x_train, y_train, x_test, y_test):
    """
    Solution for exercise 2 a)
    :param x_train: Training samples (2-dimensional)
    :param y_train: Training labels
    :param x_test: Testing samples (2-dimensional)
    :param y_test: Testing labels
    :return:
    """
    ###########
    ## TODO:
    ## Train an SVM with a linear kernel for the given dataset
    ## and plot the decision boundary and support vectors  for each using 'plot_svm_decision_boundary' function
    ###########
    SVMlin = svm.SVC(kernel='linear')
    SVMlin.fit(x_train, y_train)
    scorelin = SVMlin.score(x_test, y_test)

    plot_svm_decision_boundary(SVMlin, x_train, y_train, x_test, y_test)

    print('Linear score', scorelin)

    pass


def ex_2_b(x_train, y_train, x_test, y_test):
    """
    Solution for exercise 2 b)
    :param x_train: Training samples (2-dimensional)
    :param y_train: Training labels
    :param x_test: Testing samples (2-dimensional)
    :param y_test: Testing labels
    :return:
    """
    ###########
    ## TODO:
    ## Train SVMs with polynomial kernels for different values of the degree
    ## (Remember to set the 'coef0' parameter to 1)
    ## and plot the variation of the test and training scores with polynomial degree using 'plot_score_vs_degree' func.
    ## Plot the decision boundary and support vectors for the best value of degree
    ## using 'plot_svm_decision_boundary' function
    ###########
    degrees = range(1, 20)
    scorepolylist_test = np.zeros(np.array(degrees).shape[0])
    scorepolylist_train = np.zeros(np.array(degrees).shape[0])
    for deg in degrees:
        SVMpoly = svm.SVC(kernel='poly', coef0=1, degree=deg)
        SVMpoly.fit(x_train, y_train)
        scorepolylist_test[deg-1] = SVMpoly.score(x_test, y_test)
        scorepolylist_train[deg - 1] = SVMpoly.score(x_train, y_train)

    max_score_index = np.argmax(scorepolylist_test)
    optimal_deg = max_score_index+1
    SVMpolyopt = svm.SVC(kernel='poly', coef0=1, degree=optimal_deg)
    SVMpolyopt.fit(x_train, y_train)
    plot_score_vs_degree(scorepolylist_train, scorepolylist_test, degrees)
    plot_svm_decision_boundary(SVMpolyopt, x_train, y_train, x_test, y_test)
    print("Optimal degree", optimal_deg, "Optimal score", scorepolylist_test[max_score_index])



def ex_2_c(x_train, y_train, x_test, y_test):
    """
    Solution for exercise 2 c)
    :param x_train: Training samples (2-dimensional)
    :param y_train: Training labels
    :param x_test: Testing samples (2-dimensional)
    :param y_test: Testing labels
    :return:
    """
    ###########
    ## TODO:
    ## Train SVMs with RBF kernels for different values of the gamma
    ## and plot the variation of the test and training scores with gamma using 'plot_score_vs_gamma' function.
    ## Plot the decision boundary and support vectors for the best value of gamma
    ## using 'plot_svm_decision_boundary' function
    ###########
    gammas = np.arange(0.01, 2, 0.02)

    test_score = np.zeros(np.array(gammas).shape[0])
    train_score = np.zeros(np.array(gammas).shape[0])

    for i in range(np.array(gammas).shape[0]):
        SVMrbf = svm.SVC(kernel="rbf", gamma=gammas[i])
        SVMrbf.fit(x_train, y_train)
        test_score[i] = SVMrbf.score(x_test, y_test)
        train_score[i] = SVMrbf.score(x_train, y_train)

    max_score_index = np.argmax(test_score)
    opt_gamma = gammas[max_score_index]

    SVMrbf_opt = svm.SVC(kernel="rbf", gamma=opt_gamma)
    SVMrbf_opt.fit(x_train, y_train)
    plot_svm_decision_boundary(SVMrbf_opt, x_train, y_train, x_test, y_test)
    print("Optimal gamma", opt_gamma, "Optimal score", test_score[max_score_index])
    plot_score_vs_gamma(train_score, test_score, gammas)


def ex_3_a(x_train, y_train, x_test, y_test):
    """
    Solution for exercise 3 a)
    :param x_train: Training samples (2-dimensional)
    :param y_train: Training labels
    :param x_test: Testing samples (2-dimensional)
    :param y_test: Testing labels
    :return:
    """
    ###########
    ## TODO:
    ## Train multi-class SVMs with one-versus-rest strategy with
    ## - linear kernel
    ## - rbf kernel with gamma going from 10**-5 to 10**5
    ## - plot the scores with varying gamma using the function plot_score_versus_gamma
    ## - Mind that the chance level is not .5 anymore and add the score obtained with the linear kernel as optional argument of this function
    ###########
    SVMlin = svm.SVC(decision_function_shape='ovr', C=10,kernel='linear')
    SVMlin.fit(x_train, y_train)
    scorelin_train = SVMlin.score(x_train, y_train)
    scorelin_test = SVMlin.score(x_test, y_test)
    gammas = np.array([1e-5,1e-4,1e-3,1e-2,1e-1,1,10,1e2,1e2,1e3,1e4,1e5])
    # gammas2 = np.linspace(1e-5, 1e5, 11)
    scorerbf_train=np.zeros(np.array(gammas).shape[0])
    scorerbf_test = np.zeros(np.array(gammas).shape[0])

    for i in range(np.array(gammas).shape[0]):
        SVMrbf = svm.SVC(decision_function_shape='ovr', C=10, kernel='rbf',gamma=gammas[i])
        SVMrbf.fit(x_train, y_train)
        scorerbf_train[i]=SVMrbf.score(x_train, y_train)
        scorerbf_test[i] = SVMrbf.score(x_test, y_test)

    plot_score_vs_gamma(scorerbf_train, scorerbf_test, gamma_list=gammas, lin_score_train=scorelin_train, lin_score_test=scorelin_test)


def ex_3_b(x_train, y_train, x_test, y_test):
    """
    Solution for exercise 3 b)
    :param x_train: Training samples (2-dimensional)
    :param y_train: Training labels
    :param x_test: Testing samples (2-dimensional)
    :param y_test: Testing labels
    :return:
    """
    ###########
    ## TODO:
    ## Train multi-class SVMs with a LINEAR kernel
    ## Use the sklearn.metrics.confusion_matrix to plot the confusion matrix.
    ## Find the index for which you get the highest error rate.
    ## Plot the confusion matrix with plot_confusion_matrix.
    ## Plot the first 10 occurrences of the most misclassified digit using plot_mnist.
    ###########

    labels = range(1, 6)

    SVMlin = svm.SVC(decision_function_shape='ovr', C=10, kernel='linear')
    SVMlin.fit(x_train, y_train)
    scorelin_train = SVMlin.score(x_train, y_train)
    scorelin_test = SVMlin.score(x_test, y_test)
    y_pred = SVMlin.predict(x_test)
    conf_M = confusion_matrix(y_test, y_pred)

    most_missclass = np.argmin(np.diagonal(conf_M)) + 1


    plot_confusion_matrix(conf_M, labels)
    print(most_missclass)
    index_3 = np.where(y_test==3)
    sel_err1 = np.array([0])  # Numpy indices to select images that are misclassified.
    sel_err = np.array([0])  # Numpy indices to select images that are misclassified.
    sel_err1 = y_pred[(y_pred[index_3] - y_test[index_3] != 0) == True]
    print(sel_err1)
    sel_err = index_3[np.asarray(sel_err1)]

    print(index_3, np.where(y_pred[index_3] - y_test[index_3] != 0))
    i = most_missclass  # should be the label number corresponding the largest classification error

    # Plot with mnist plot
    plot_mnist(x_test[sel_err], y_pred[sel_err], labels=labels[i], k_plots=10, prefix='Predicted class')
