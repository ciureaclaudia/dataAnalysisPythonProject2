import pandas as pd

from functii import *
from sklearn.decomposition import PCA

tabel_date = pd.read_csv("setDate.csv", index_col=0, low_memory=False)

print("nr linii: ", tabel_date.shape[0], "  nr col: ", tabel_date.shape[1])
# print(tabel_date.values)  # valorile din tabel_date

variabile = list(tabel_date)[1:]  # lista cu cap de tabel
x = tabel_date[variabile].values  # matrice cu toate valorile pe fiecare linie
dist, k_opt = calcul_ierarhie(x)  # k_opt nr de clusteri din partitia optimala

# print(max(dist[:, 1]))
# plot_ierarhie(dist, tabel_date.index, max(dist[:, 1]) + 1)
# show()

# Creez partitia optimala
partitie_optimala = calcul_partitie(x, k_opt)  #partitie_optimala intoarce un array: partitie_optimala[0] si un threshold: partitie_optimala[1]
plot_ierarhie(dist, tabel_date.index, partitie_optimala[1], "Plot partitie optimala")  # desenez partitia: DENDOGRAMA
print('Componența partieiei optimale: ',partitie_optimala[0]) #componenta partitiei
show()

# CALCULEZ AXELE PRINCIPALE  -> pentru trasare plot partitie
pca=PCA(2)
pca.fit(x)
z=pca.transform(x)
#trasez plotul cu axele calculate mai sus
plot_partitie(z,partitie_optimala[0]," Optimal",tabel_date.index)
show()

# Desenare histograme
for i in range(len(variabile)):
    histograme(x[:, i], partitie_optimala[0], variabile[i])
show()


# creez partitie din 3 clusteri
partitie3 = calcul_partitie(x, 3)
plot_ierarhie(dist, tabel_date.index, partitie3[1], "Plot partitie din 3 clusteri")  # desenez partitia
print(partitie3[0])
show()

# CALCULEZ AXELE PRINCIPALE  -> pentru trasare plot partitie
pca=PCA(2)
pca.fit(x)
z=pca.transform(x)
plot_partitie(z,partitie3[0]," Partitie 3 clusteri",tabel_date.index) #desenez plotul cu axele calculate mai sus
show()

# Desenare histograme
for i in range(len(variabile)):
    histograme(x[:, i], partitie3[0], variabile[i])
show()

# # creez partitie din 2 clusteri
# partitie2 = calcul_partitie(x, 2)
# plot_ierarhie(dist, tabel_date.index, partitie2[1], "Plot partitie din 2 clusteri")  # desenez partitia
# print(partitie2[0])
# show()
#
# # Desenare histograme
# for i in range(len(variabile)):
#     histograme(x[:, i], partitie3[0], variabile[i])
# show()

# Salvam partitiile intr un tabel
tabel_partitii = pd.DataFrame(
    data={
        "Partitieoptimala": partitie_optimala[0],
        "Partitie3clusteri": partitie3[0],
        # "Partitie2clusteri": partitie2[0],
    }, index=tabel_date.index
)
# print(tabel_partitii)
tabel_partitii.to_csv("partitii.csv")




