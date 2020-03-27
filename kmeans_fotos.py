import glob
import numpy as np
import matplotlib.pylab as plt
import sklearn.cluster
archivos=glob.glob('imagenes/*.png')

datos=[]
for i in archivos:
    data = plt.imread(i)
    X = data.reshape((-1))
    datos.append(X)
    
n_clusters=np.arange(1,21)
d=[]
for i in n_clusters:
    k_means = sklearn.cluster.KMeans(n_clusters=i)
    k_means.fit(datos)
    d.append(k_means.inertia_)

plt.plot(n_clusters,d)
plt.ylabel('Inercia')
plt.xlabel('Número de Clusters')
plt.title('El Mejor número de Clusters es 4')
plt.savefig('inercia.png')
plt.close()

#Clusters
k_f = sklearn.cluster.KMeans(n_clusters=4)
k_f.fit(datos)
centros=k_f.cluster_centers_
cluster = k_f.predict(datos)
def distancias(n):
    centro=centros[n]
    select= np.where(cluster==n)
    puntos=np.array(datos)[select,:]
    nombres=np.array(archivos)[select]
    dist=[]
    for i in puntos[0]:
        dist.append(np.linalg.norm(centro-i))
    ot=dist.copy()
    ot.sort()
    may=ot[:5]
    ind=[]
    for i in may:
        index=np.where(np.array(dist)==i)
        ind.append(index[0][0])
    return nombres[ind]
cerc=[]
clus=np.arange(1,5)
for i in clus:
    cerc.append(distancias(i-1))

#Gráfica de los Clusters

plt.figure(figsize=(18,18))
j=np.ones(20,dtype=int)
j[:5]=0
j[5:10]=1
j[10:15]=2
j[15:20]=3
for i in range(20):
    plt.subplot(4,5,i+1)
    df=plt.imread(cerc[j[i]][i%5])
    plt.imshow(df)
    plt.title('Clúster #'+str(j[i]+1))
plt.savefig('ejemplo_clases.png')