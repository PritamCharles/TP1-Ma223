import matplotlib.pyplot as plt
import itertools


class Chart:
    def __init__(self, all_in_one, xlabel=None, ylabel=None, purpose=None, title=None):
        self.all_in_one = all_in_one
        self.purpose = purpose
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel

    def plot(self, list_xvalues1, list_yvalues1, list_labels, nb_plots=1, list_xvalues2=None, list_yvalues2=None, title_list=None, axis_labels=None):
        plt.figure(figsize=(15, 9))

        if self.all_in_one:
            plt.subplot(2, 1, 1)
            for i, j, k, l in itertools.zip_longest(range(nb_plots - 1), range(nb_plots - 1), range(nb_plots - 1),
                                                    range(nb_plots - 1), fillvalue=""):
                plt.plot(list_xvalues1[j], list_yvalues1[k], label=list_labels[l])
                plt.title("Temps d'execution des differentes methodes en fonction de la taille de la matrice")
            plt.legend()
            plt.grid()

            plt.subplot(2, 1, 2)
            for i, j, k, l in itertools.zip_longest(range(nb_plots - 1), range(nb_plots - 1), range(nb_plots - 1),
                                                    range(nb_plots - 1), fillvalue=""):
                plt.plot(list_xvalues2[j], list_yvalues2[k], label=list_labels[l])
                plt.title("Normes des differentes methodes en fonction de la taille de la matrice")
            plt.legend()
            plt.grid()

        elif not self.all_in_one:
            plt.subplot(2, 1, 2)
            plt.plot(list_xvalues1, list_yvalues1, label=list_labels)
            plt.title(title_list[0])
            plt.xlabel(axis_labels[0])
            plt.ylabel(axis_labels[1])
            plt.legend()
            plt.grid()

            plt.subplot(2, 1, 2)
            plt.plot(list_xvalues1, list_yvalues1, label=list_labels)
            plt.title(title_list[1])
            plt.xlabel(axis_labels[0])
            plt.ylabel(axis_labels[2])
            plt.legend()
            plt.grid()

        plt.show()
