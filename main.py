from src.gauss_method import GaussMethod
from src.lu_method import LUMethod
from src.charts import Chart
import numpy as np
import time


class GetPlots:
    def __init__(self, array_minsize, array_maxsize, step):
        self.gauss = GaussMethod()
        self.LU = LUMethod()
        self.array_minsize = array_minsize
        self.array_maxsize = array_maxsize
        self.step = step

    def get_xplots(self):
        return [i for i in range(self.array_minsize, self.array_maxsize, self.step)]

    def get_yplots_gauss(self):
        timeslist_gauss, normslist_gauss = [], []

        for i in range(self.array_minsize, self.array_maxsize, self.step):
            A = np.random.rand(i, i)
            B = np.random.rand(i, 1)

            start = time.time()
            X = self.gauss.Gauss(A, B)
            end = time.time()
            exectime = end - start
            timeslist_gauss.append(exectime)

            norm = np.linalg.norm(np.dot(A, X) - np.ravel(B))
            normslist_gauss.append(norm)

        return timeslist_gauss, normslist_gauss

    def get_yplots_pp(self):
        timeslist_pp, normslist_pp = [], []

        for i in range(self.array_minsize, self.array_maxsize, self.step):
            A = np.random.rand(i, i)
            B = np.random.rand(i, 1)

            start = time.time()
            X = self.gauss.GaussPartialPivotChoice(A, B)
            end = time.time()
            exectime = end - start
            timeslist_pp.append(exectime)

            norm = np.linalg.norm(np.dot(A, X) - np.ravel(B))
            normslist_pp.append(norm)

        return timeslist_pp, normslist_pp

    def get_yplots_tp(self):
        timeslist_tp, normslist_tp = [], []

        for i in range(self.array_minsize, self.array_maxsize, self.step):
            A = np.random.rand(i, i)
            B = np.random.rand(i, 1)

            start = time.time()
            X = self.gauss.GaussTotalPivotChoice(A, B)
            end = time.time()
            exectime = end - start
            timeslist_tp.append(exectime)

            norm = np.linalg.norm(np.dot(A, X) - np.ravel(B))
            normslist_tp.append(norm)

        return timeslist_tp, normslist_tp

    def get_yplots_lu(self):
        timeslist_lu, normslist_lu = [], []

        for i in range(self.array_minsize, self.array_maxsize, self.step):
            A = np.random.rand(i, i)
            B = np.random.rand(i, 1)

            L = self.LU.LUDecomposition(A)[0]
            U = self.LU.LUDecomposition(A)[1]
            start = time.time()
            X = self.LU.LUResolution(L, U, B)
            end = time.time()
            exectime = end - start
            timeslist_lu.append(exectime)

            A = np.dot(L, U)
            norm = np.linalg.norm(np.dot(A, X) - np.ravel(B))
            normslist_lu.append(norm)

        return timeslist_lu, normslist_lu

    def get_yplots_np(self):
        timeslist_npsolve, normslist_npsolve = [], []

        for i in range(self.array_minsize, self.array_maxsize, self.step):
            A = np.random.rand(i, i)
            B = np.random.rand(i, 1)

            start = time.time()
            X = np.linalg.solve(A, B)
            end = time.time()
            exectime = end - start
            timeslist_npsolve.append(exectime)

            norm = np.linalg.norm(np.dot(A, X) - B)
            normslist_npsolve.append(norm)

        return timeslist_npsolve, normslist_npsolve


###
plots = GetPlots(array_minsize=3, array_maxsize=353, step=50)

xlist = [plots.get_xplots(), plots.get_xplots(), plots.get_xplots(), plots.get_xplots(), plots.get_xplots()]
yplots_gauss = [plots.get_yplots_gauss()[0], plots.get_yplots_gauss()[1]]
yplots_pp = [plots.get_yplots_pp()[0], plots.get_yplots_pp()[1]]
yplots_tp = [plots.get_yplots_tp()[0], plots.get_yplots_tp()[1]]
yplots_lu = [plots.get_yplots_lu()[0], plots.get_yplots_lu()[1]]
yplots_np = [plots.get_yplots_np()[0], plots.get_yplots_np()[1]]
ytimeslist = [yplots_gauss[0], yplots_pp[0], yplots_tp[0], yplots_lu[0], yplots_np[0]]
ynormslist = [yplots_gauss[1], yplots_pp[1], yplots_tp[1], yplots_lu[1], yplots_np[1]]

labels_list = ["Gauss", "Gauss Pivot partiel", "Gauss Pivot total", "LU", "np.linalg.solve"]
titles_list_gauss = ["Temps d'execution de la methode de Gauss en fonction de la taille de la matrice", "Erreur ||AX - B|| de la methode de Gauss en fonction de la taille de la matrice"]
titles_list_pp = ["Temps d'execution de la methode du pivot partiel en fonction de la taille de la matrice", "Erreur ||AX - B|| de la methode du pivot partiel en fonction de la taille de la matrice"]
titles_list_tp = ["Temps d'execution de la methode du pivot total en fonction de la taille de la matrice", "Erreur ||AX - B|| de la methode du pivot total en fonction de la taille de la matrice"]
titles_list_lu = ["Temps d'execution de la methode de LU en fonction de la taille de la matrice", "Erreur ||AX - B|| de la methode LU en fonction de la taille de la matrice"]
titles_list_np = ["Temps d'execution de la methode linalg.solve de Numpy en fonction de la taille de la matrice", "Erreur ||AX - B|| de la methode linalg.solve de Numpy en fonction de la taille de la matrice"]
titles_list_all = ["Temps d'execution des differentes methodes en fonction de la taille de la matrice", "Erreur ||AX - B|| des differentes methodes en fonction de la taille de la matrice"]
axis_labels = ["Taille de la matrice n", "Temps d'execution (en s)", "Erreur ||AX - B||", "log Taille de la matrice n", "log Temps d'execution (en s)"]

# Solo charts
chart1 = Chart(title=titles_list_gauss, alabels=axis_labels)
chart2 = Chart(title=titles_list_pp, alabels=axis_labels)
chart3 = Chart(title=titles_list_tp, alabels=axis_labels)
chart4 = Chart(title=titles_list_lu, alabels=axis_labels)
chart5 = Chart(title=titles_list_np, alabels=axis_labels)

chart1.plot(list_xvalues1=xlist[0], list_yvalues1=ytimeslist[0], list_xvalues2=xlist[0], list_yvalues2=ynormslist[0], list_labels=labels_list[0])
chart1.plot_log(list_xvalues=np.log(xlist[0]), list_yvalues=np.log(ytimeslist[0]), list_labels=labels_list[0])

chart2.plot(list_xvalues1=xlist[0], list_yvalues1=ytimeslist[1], list_xvalues2=xlist[0], list_yvalues2=ynormslist[1], list_labels=labels_list[1])
chart2.plot_log(list_xvalues=np.log(xlist[0]), list_yvalues=np.log(ytimeslist[0]), list_labels=labels_list[1])

chart3.plot(list_xvalues1=xlist[0], list_yvalues1=ytimeslist[2], list_xvalues2=xlist[0], list_yvalues2=ynormslist[2], list_labels=labels_list[2])
chart3.plot_log(list_xvalues=np.log(xlist[0]), list_yvalues=np.log(ytimeslist[0]), list_labels=labels_list[2])

chart4.plot(list_xvalues1=xlist[0], list_yvalues1=ytimeslist[3], list_xvalues2=xlist[0], list_yvalues2=ynormslist[3], list_labels=labels_list[3])
chart4.plot_log(list_xvalues=np.log(xlist[0]), list_yvalues=np.log(ytimeslist[0]), list_labels=labels_list[3])

chart5.plot(list_xvalues1=xlist[0], list_yvalues1=ytimeslist[4], list_xvalues2=xlist[0], list_yvalues2=ynormslist[4], list_labels=labels_list[4])
chart5.plot_log(list_xvalues=np.log(xlist[0]), list_yvalues=np.log(ytimeslist[0]), list_labels=labels_list[4])


# All in one chart
chart6 = Chart(title=titles_list_all, alabels=axis_labels)
chart6.plot_all(nb_plots=5, list_xvalues1=xlist, list_yvalues1=ytimeslist, list_xvalues2=xlist, list_yvalues2=ynormslist, list_labels=labels_list)
