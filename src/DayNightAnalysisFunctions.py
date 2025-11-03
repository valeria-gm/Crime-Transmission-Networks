import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr
from sklearn.metrics import jaccard_score

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio

def compararElementos(A,B):
    m,n = A.shape
    diferencias = []
    for i in range(m):
        for j in range(n):
            if A[i,j] != B[i,j]:
                diferencias.append((i,j))
    return diferencias

def ind2group(L):
    g = list(set(L))
    d = dict() 
    for gIx in g:
        d[int(gIx)] = [i for i,l in enumerate(L) if l==gIx]
    return d        

def writeClusterResults(E,groups,rPath):
    for ix,indices in groups.items():
        fname = rPath + 'group_'+str(ix)+'.txt'
        np.savetxt(fname,E[indices],fmt='%.0f')       
    return

def radar_simple(A, D, P, titulo="Dinámica Criminal"):
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=[A, D, P],
        theta=['Activation', 'Deactivation', 'Persistence'],
        fill='toself',
        name=titulo,
        line_color='blue'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        title=titulo,
        font_size=12
    )
    
    return fig

def radar_multiple(datos_crimenes, nombres):
    fig = go.Figure()
    
    for i, (A, D, P) in enumerate(datos_crimenes):
        fig.add_trace(go.Scatterpolar(
            r=[A, D, P],
            theta=['Activation', 'Deactivation', 'Persistence'],
            fill='toself',
            name=nombres[i]
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        title="Comparison of criminal dynamics",
        font_size=12
    )
    
    return fig

def getMatrix(tipo, horario,i_delito):

    workPath = '/home/hp/Documentos/TICS/Tesis/'
    
    delitos = ['RoboCasa','RoboComercios','RoboVehiculos']
    dataPaths = ['Resultados_RoboCasa/',
             'Resultados_RoboComercios/',
             'Resultados_RoboVehiculos/']
    delito   = delitos[i_delito]
    dataPath = workPath  + dataPaths[i_delito]
    
    if horario == 'd':
        prefijo = '_Dia_'
    elif horario == 'n':
        prefijo = '_Noche_'
    else:
        prefijo = '_Gen_'
     
    if tipo == 'obs':
        sufijo = 'matriz_adyacencia_obs.txt'
    elif tipo == 'base' or tipo == 'zone':
        sufijo = 'significancia_sim.txt'
    elif tipo == 'out':
        sufijo = 'significancia_outflow.txt'
    elif tipo == 'in':
        sufijo = 'significancia_inflow.txt'
    
    
    fileName = dataPath+delito+prefijo+sufijo
    A = np.loadtxt(fileName)
    
    if tipo == 'zone':
        A = np.diag(A)
    
    return A   

def getStateMatrix( horario, i_delito):
    a = getMatrix('in', horario,i_delito)
    b = getMatrix('out', horario,i_delito)
    c = getMatrix('zone', horario,i_delito)
    
    M = np.vstack((a,b,c))
    return M   

def translateStates(M):
    stateDict = {(0,0,0):0,
                 (0,0,1):1,
                 (0,1,0):2,
                 (0,1,1):3,
                 (1,0,0):4,
                 (1,0,1):5,
                 (1,1,0):6,
                 (1,1,1):7}
    m,n = M.shape
    S   = np.zeros(n)
    
    for i in range(n):
        #state = tuple(M[:,i])
        state = tuple(int(round(x)) for x in M[:, i])
        S[i]  = stateDict[state]
    return S

def getTransitionMatrix(S1,S2):
    M = np.zeros((8,8))
    
    n = len(S1)
    for ix in range(n):
        i = int(S1[ix])
        j = int(S2[ix])
        M[i,j] += 1
    
    return M   

def getIndicesTotal(T):
    N = np.sum(T)
    
    # stability
    S = np.trace(T)/N
    
    # dynamic crime switch
    D = (np.sum(T[0,1:]) + np.sum(T[1:,0]))/N
        
    
    # persistence
    Ta = T[1:,1:]
    P  = np.sum(Ta)/N
    
    # urban crime dynamic index
    UCDI = np.mean([S,D,P])
    
    return S, D, P, UCDI     


def getIndices(T):
    n = np.sum(T)
    N = n - T[0, 0] if n > T[0, 0] else n
    Ta = T[1:, 1:]

    A = np.sum(T[0, 1:]) / N
    D = np.sum(T[1:, 0]) / N
    P = np.sum(Ta) / N
    IR = (A + D) / N

    return A, D, P, IR

def getPatterns(A,thres=.1):
    tVal = np.round(thres*(np.sum(A)-A[0,0]))
    P    = np.argwhere(A>=tVal)    
    return P

def createEmbedding(S1,S2,A1,A2):
    N = len(S1)
    E = np.zeros((N,4))
    E[:,0] = S1
    E[:,1] = S2
    for i in range(N):
        flujo1  = np.sum(A1[i,:])+np.sum(A1[:,i]) - 2*A1[i,i]
        E[i,2] = flujo1
        flujo2  = np.sum(A2[i,:])+np.sum(A2[:,i]) - 2*A2[i,i]
        E[i,3] = flujo2
    
    return E    

def distCat(a,b):
    if a == b:
        dcat = 0
    else:
        if (a == 0) or (b == 0):
            dcat = 2
        else:
            dcat = 1
    return dcat

def distZona(z1,z2,rangoFlujo1=1,rangoFlujo2=1,w=0.7):
    # distancia categórica
    dcat = distCat(z1[0],z2[0]) + distCat(z1[1],z2[1])
    dnum = np.abs(z1[2]-z2[2])/rangoFlujo1 + np.abs(z1[3]-z2[3])/rangoFlujo2 
    d = w*dcat + (1-w)*dnum
    return d

def getDistanceMatrix(E):
    m,n = E.shape
    D   = np.zeros((m,m))
    rango1 = max(E[:,2]) - min(E[:,2])
    rango2 = max(E[:,3]) - min(E[:,3])

    for i in range(m-1):
        for j in range(i+1,m):
            d = distZona(E[i,:],E[j,:],rango1,rango2,0.75)
            D[i,j] = d
            D[j,i] = d
    return D        

def graficar_agebs_por_cluster(gdf_agebs, clusters_array, crimen):
  
    if len(gdf_agebs) != len(clusters_array):
        raise ValueError("El array de clústeres no coincide con el número de AGEBs.")

    gdf_agebs = gdf_agebs.copy()
    gdf_agebs['cluster'] = clusters_array

    num_clusters = len(np.unique(clusters_array))
    cmap = plt.get_cmap('Pastel1' if num_clusters <= 9 else 'tab20c')
    colors = [cmap(i) for i in range(num_clusters)]
    color_dict = {cluster: colors[i] for i, cluster in enumerate(np.unique(clusters_array))}

    # Asignar colores según el clúster
    gdf_agebs['color'] = gdf_agebs['cluster'].map(color_dict)

    fig, ax = plt.subplots(figsize=(10, 10))
    gdf_agebs.plot(ax=ax, color=gdf_agebs['color'], edgecolor='black', linewidth=0.5)

    legend_patches = [
        plt.Line2D([0], [0], marker='o', color='w', label=f'Cluster {cluster}',
                   markerfacecolor=color_dict[cluster], markersize=10)
        for cluster in sorted(np.unique(clusters_array))
    ]
    ax.legend(handles=legend_patches, loc='upper right')

    titulo = f"Hierarchal clustering for {crimen}"
    ax.set_title(titulo)
    ax.set_xlabel("Longitude (EPSG:32614)")
    ax.set_ylabel("Latitude (EPSG:32614)")
    plt.axis('equal')
    plt.tight_layout()
    nombre_archivo = f"AGEBS_clusters_{crimen.lower().replace(' ', '_')}.png"
    plt.savefig(nombre_archivo, dpi=1200, bbox_inches='tight')  # Guardar imagen
    plt.show()