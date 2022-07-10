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

    def get_yplots_times(self):
        timeslist_gauss, timeslist_pp, timeslist_tp, timeslist_lu, timeslist_npsolve = [], [], [], [], []

        for i in range(self.array_minsize, self.array_maxsize, self.step):
            A = np.random.rand(i, i)
            B = np.random.rand(i, 1)

            start = time.time()
            self.gauss.Gauss(A, B)
            end = time.time()
            exectime = end - start
            timeslist_gauss.append(exectime)

            start = time.time()
            self.gauss.GaussPartialPivotChoice(A, B)
            end = time.time()
            exectime = end - start
            timeslist_pp.append(exectime)

            start = time.time()
            self.gauss.GaussTotalPivotChoice(A, B)
            end = time.time()
            exectime = end - start
            timeslist_tp.append(exectime)

            L = self.LU.LUDecomposition(A)[0]
            U = self.LU.LUDecomposition(A)[1]
            start = time.time()
            self.LU.LUResolution(L, U, B)
            end = time.time()
            exectime = end - start
            timeslist_lu.append(exectime)

            start = time.time()
            np.linalg.solve(A, B)
            end = time.time()
            exectime = end - start
            timeslist_npsolve.append(exectime)

        return timeslist_gauss, timeslist_pp, timeslist_tp, timeslist_lu, timeslist_npsolve

    def get_yplots_norms(self):
        normslist_gauss, normslist_pp, normslist_tp, normslist_lu, normslist_npsolve = [], [], [], [], []

        for i in range(self.array_minsize, self.array_maxsize, self.step):
            A = np.random.rand(i, i)
            B = np.random.rand(i, 1)

            X = self.gauss.Gauss(A, B)
            norm = np.linalg.norm(np.dot(A, X) - np.ravel(B))
            normslist_gauss.append(norm)

            X = self.gauss.GaussPartialPivotChoice(A, B)
            norm = np.linalg.norm(np.dot(A, X) - np.ravel(B))
            normslist_pp.append(norm)

            X = self.gauss.GaussTotalPivotChoice(A, B)
            norm = np.linalg.norm(np.dot(A, X) - np.ravel(B))
            normslist_tp.append(norm)

            L = self.LU.LUDecomposition(A)[0]
            U = self.LU.LUDecomposition(A)[1]
            X = self.LU.LUResolution(L, U, B)
            A = np.dot(L, U)
            norm = np.linalg.norm(np.dot(A, X) - np.ravel(B))
            normslist_lu.append(norm)

            X = np.linalg.solve(A, B)
            norm = np.linalg.norm(np.dot(A, X) - B)
            normslist_npsolve.append(norm)

        return normslist_gauss, normslist_pp, normslist_tp, normslist_lu, normslist_npsolve

###
# All in one charts
chart100 = Chart(all_in_one=True, purpose="time", xlabel="Taille de la matrice n", ylabel="Temps d'exécution (en s)")
plots = GetPlots(array_minsize=3, array_maxsize=203, step=50)

xlist = [plots.get_xplots(), plots.get_xplots(), plots.get_xplots(), plots.get_xplots(), plots.get_xplots()]
ytimeslist = [plots.get_yplots_times()[0], plots.get_yplots_times()[1], plots.get_yplots_times()[3], plots.get_yplots_times()[4]]
ynormslist = [plots.get_yplots_norms()[0], plots.get_yplots_norms()[1], plots.get_yplots_norms()[3], plots.get_yplots_norms()[4]]
labelslist = ["Gauss", "Gauss Pivot partiel,", "Gauss Pivot total", "LU", "np.linalg.solve"]

#chart100.plot(nb_plots=5, list_xvalues1=xlist, list_yvalues1=ytimeslist, list_xvalues2=xlist, list_yvalues2=ynormslist, list_labels=labelslist)

# Solo methods charts
title_list1 = ["Temps d'execution de la methode de Gauss en fonction de la taille de la matrice", "Erreur ||AX -B|| de la methode de Gauss en fonction de la taille de la matrice"]
title_list2 = ["Temps d'execution de la methode du pivot partiel en fonction de la taille de la matrice", "Erreur ||AX -B|| de la methode du pivot partiel en fonction de la taille de la matrice"]
title_list3 = ["Temps d'execution de la methode du pivot total en fonction de la taille de la matrice", "Erreur ||AX -B|| de la methode du pivot total en fonction de la taille de la matrice"]
title_list4 = ["Temps d'execution de la methode de LU en fonction de la taille de la matrice", "Erreur ||AX -B|| de la methode de LU en fonction de la taille de la matrice"]
title_list5 = ["Temps d'execution de la methode linalg.solve de Numpy en fonction de la taille de la matrice", "Erreur ||AX -B|| de la methode linalg.solve de Numpy en fonction de la taille de la matrice"]
axis_labels = ["Taille de la matrice n", "Temps d'execution (en s)", "Erreur ||AX - B||"]

chart1 = Chart(all_in_one=False, title=title_list1, xlabel=None, ylabel=None)
chart2 = Chart(all_in_one=False, title=title_list2, xlabel=None, ylabel=None)
chart3 = Chart(all_in_one=False, title=title_list3, xlabel=None, ylabel=None)
chart4 = Chart(all_in_one=False, title=title_list4, xlabel=None, ylabel=None)
chart5 = Chart(all_in_one=False, title=title_list5, xlabel=None, ylabel=None)

chart1.plot(list_xvalues1=xlist, list_yvalues1=ytimeslist[0], list_xvalues2=xlist, list_yvalues2=ynormslist[0], list_labels=labelslist)
#chart2.plot(list_xvalues1=xlist, list_yvalues1=ytimeslist[0], list_xvalues2=xlist, list_yvalues2=ynormslist[0], list_labels=labelslist)
#chart3.plot(list_xvalues1=xlist, list_yvalues1=ytimeslist[0], list_xvalues2=xlist, list_yvalues2=ynormslist[0], list_labels=labelslist)
#chart4.plot(list_xvalues1=xlist, list_yvalues1=ytimeslist[0], list_xvalues2=xlist, list_yvalues2=ynormslist[0], list_labels=labelslist)
#chart5.plot(list_xvalues1=xlist, list_yvalues1=ytimeslist[0], list_xvalues2=xlist, list_yvalues2=ynormslist[0], list_labels=labelslist)
