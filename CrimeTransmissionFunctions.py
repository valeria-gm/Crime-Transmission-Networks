import pandas as pd
import geopandas as gpd
import networkx as nx
import numpy as np
import random
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from shapely.geometry import Point
from collections import Counter


def redDelitoDirigida(df_Delito, delta_t, delta_s):
    G_delito = nx.DiGraph()
    assert 'geometry' in df_Delito.columns, "The DataFrame must have a 'geometry' column."
    assert 'FechaComision' in df_Delito.columns, "The DataFrame must have a 'FechaComision' column."
    df_Delito['FechaComision'] = pd.to_datetime(df_Delito['FechaComision'])
    coords = np.array(list(zip(df_Delito.geometry.x, df_Delito.geometry.y)))
    dates = df_Delito['FechaComision'].values  
    n = len(df_Delito)
    if n == 0:
        return G_delito
    dist_matrix = cdist(coords, coords, metric='euclidean')
    day_diff = (dates[None, :] - dates[:, None]) / np.timedelta64(1, 'D')
    time_mask = (day_diff > 0) & (day_diff <= delta_t)  
    spatial_mask = (dist_matrix <= delta_s)             
    final_mask = time_mask & spatial_mask              
    origin_idx, dest_idx = np.where(final_mask)
    for i, j in zip(origin_idx, dest_idx):
        if i not in G_delito:
            G_delito.add_node(i, lat=coords[i, 1], lon=coords[i, 0], fecha=dates[i])
        if j not in G_delito:
            G_delito.add_node(j, lat=coords[j, 1], lon=coords[j, 0], fecha=dates[j])
        distancia = dist_matrix[i, j]
        diferencia_dias = day_diff[i, j]
        G_delito.add_edge(i, j, distancia=distancia, dias=diferencia_dias)
    nodos_aislados = list(nx.isolates(G_delito))  
    G_delito.remove_nodes_from(nodos_aislados)    
    print("Number of connected nodes:", G_delito.number_of_nodes())
    print("Number of edges:", G_delito.number_of_edges())
    return G_delito


def cargar_agebs(filepath):
    gdf_agebs = gpd.read_file(filepath)
    # Morelia: 053
    gdf_agebs = gdf_agebs[gdf_agebs['CVE_MUN'] == '053'].reset_index(drop=True)
    # CRS EPSG:32614 (UTM Zona 14N)
    if gdf_agebs.crs is None or gdf_agebs.crs.to_string() != "EPSG:32614":
        gdf_agebs = gdf_agebs.to_crs("EPSG:32614")
    return gdf_agebs


def asignar_nombres_agebs(gdf_agebs, G_delito):
    nodos_agebs = {}
    agebs_nombres = {idx: f"AGEB_{idx}" for idx in gdf_agebs.index}
    gdf_agebs["centro"] = gdf_agebs.geometry.centroid
    for nodo, data in G_delito.nodes(data=True):
        punto = Point(data['lon'], data['lat']) 
        agebs_tocando = gdf_agebs[gdf_agebs.geometry.contains(punto)]
        if agebs_tocando.empty:
            agebs_tocando = gdf_agebs[gdf_agebs.geometry.touches(punto)]
        if not agebs_tocando.empty:
            if len(agebs_tocando) > 1:
                agebs_tocando["distancia"] = agebs_tocando["centro"].apply(lambda centro: punto.distance(centro))
                ageb_asignado = agebs_tocando.sort_values("distancia").iloc[0]
            else:
                ageb_asignado = agebs_tocando.iloc[0]
            nodos_agebs[nodo] = agebs_nombres[ageb_asignado.name]
    for idx, nombre in agebs_nombres.items():
        if nombre not in nodos_agebs.values():
            nodos_agebs[f"dummy_{idx}"] = nombre
    return nodos_agebs, agebs_nombres

    
def crear_matriz_adyacencia_agebs(G_delito, nodos_agebs, agebs_nombres):
    n_agebs = len(agebs_nombres)
    matriz = np.zeros((n_agebs, n_agebs), dtype=int)
    ageb_indices = {nombre: idx for idx, nombre in agebs_nombres.items()}
    for nodo_origen, nodo_destino in G_delito.edges():
        if nodo_origen in nodos_agebs and nodo_destino in nodos_agebs: 
            ageb_origen = nodos_agebs[nodo_origen]
            ageb_destino = nodos_agebs[nodo_destino]
            idx_origen = ageb_indices[ageb_origen]
            idx_destino = ageb_indices[ageb_destino]
            matriz[idx_origen, idx_destino] += 1
    return matriz


def monte_carlo_simulaciones_agebs(gdf_Crimen, agebs_gdf, n_simulaciones=99, seed=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    target_crs = "EPSG:32614"
    gdf_Crimen = gdf_Crimen.to_crs(target_crs)
    agebs_gdf = agebs_gdf.to_crs(target_crs)
    minx, miny, maxx, maxy = agebs_gdf.total_bounds
    simulaciones = []
    for _ in range(n_simulaciones):
        n_crimenes = len(gdf_Crimen)
        extra_ratio = 1.5
        n_puntos = int(n_crimenes * extra_ratio)
        xs = np.random.uniform(minx, maxx, n_puntos)
        ys = np.random.uniform(miny, maxy, n_puntos)
        puntos = gpd.GeoDataFrame(geometry=[Point(x, y) for x, y in zip(xs, ys)], crs=target_crs)
        puntos_validos = gpd.sjoin(puntos, agebs_gdf, how="inner", predicate="within").drop(columns="index_right")
        while len(puntos_validos) < n_crimenes:
            faltan = n_crimenes - len(puntos_validos)
            xs = np.random.uniform(minx, maxx, int(faltan * extra_ratio))
            ys = np.random.uniform(miny, maxy, int(faltan * extra_ratio))
            nuevos_puntos = gpd.GeoDataFrame(geometry=[Point(x, y) for x, y in zip(xs, ys)], crs=target_crs)
            nuevos_validos = gpd.sjoin(nuevos_puntos, agebs_gdf, how="inner", predicate="within").drop(columns="index_right")
            puntos_validos = pd.concat([puntos_validos, nuevos_validos], ignore_index=True)
        puntos_finales = puntos_validos.sample(n=n_crimenes, random_state=seed).reset_index(drop=True)
        fechas_revueltas = np.random.permutation(gdf_Crimen['FechaComision'].values)
        simulacion = gpd.GeoDataFrame({
            'FechaComision': fechas_revueltas,
            'Delito': gdf_Crimen['Delito'].values,
            'geometry': puntos_finales.geometry
        }, crs=target_crs)
        simulaciones.append(simulacion)
    return simulaciones


def procesar_simulacion(gdf_simulacion, gdf_agebs, delta_t=7, delta_s=500):
    df_joined = gpd.sjoin(gdf_simulacion, gdf_agebs[['geometry']], how='inner', predicate='within')
    df_joined['AGEB'] = df_joined['index_right'].astype(int)
    if df_joined.empty:
        return np.zeros((len(gdf_agebs), len(gdf_agebs)), dtype=int)
    coords = np.vstack([df_joined.geometry.x.values, df_joined.geometry.y.values]).T
    df_joined['FechaComision'] = pd.to_datetime(df_joined['FechaComision'])
    dates = df_joined['FechaComision'].values
    dist_matrix = cdist(coords, coords) 
    day_diff = (dates[None, :] - dates[:, None]) / np.timedelta64(1, 'D')
    final_mask = (day_diff > 0) & (day_diff <= delta_t) & (dist_matrix <= delta_s)
    sorted_agebs = gdf_agebs.index.sort_values()
    ageb_to_index = {ageb: idx for idx, ageb in enumerate(sorted_agebs)}
    n_agebs = len(ageb_to_index)
    matrix = np.zeros((n_agebs, n_agebs), dtype=int)
    origin_idx, dest_idx = np.where(final_mask)
    ageb_array = df_joined['AGEB'].values
    origin_agebs = ageb_array[origin_idx]
    dest_agebs = ageb_array[dest_idx]
    non_self = origin_agebs != dest_agebs
    if np.any(non_self):
        origin_idx_mapped = np.fromiter((ageb_to_index[ageb] for ageb in origin_agebs[non_self]), dtype=int)
        dest_idx_mapped = np.fromiter((ageb_to_index[ageb] for ageb in dest_agebs[non_self]), dtype=int)
        np.add.at(matrix, (origin_idx_mapped, dest_idx_mapped), 1)
    same_ageb = origin_agebs == dest_agebs
    if np.any(same_ageb):
        diag_idx_mapped = np.fromiter((ageb_to_index[ageb] for ageb in origin_agebs[same_ageb]), dtype=int)
        np.add.at(matrix, (diag_idx_mapped, diag_idx_mapped), 1)
    return matrix


def procesar_simulaciones(simulaciones, gdf_agebs, delta_t=7, delta_s=500):
    lista_matrices = [procesar_simulacion(sim, gdf_agebs, delta_t, delta_s) for sim in simulaciones]
    matriz_promedio = np.mean(lista_matrices, axis=0)
    return matriz_promedio, lista_matrices


def matriz_a_grafo(matriz, gdf_agebs):
    G = nx.DiGraph()
    sorted_agebs = gdf_agebs.index.sort_values()
    for ageb in sorted_agebs:
        centroide = gdf_agebs.loc[ageb, 'geometry'].centroid
        G.add_node(ageb, label=f"AGEB_{ageb}", lon=centroide.x, lat=centroide.y)
    for i, origin in enumerate(sorted_agebs):
        for j, dest in enumerate(sorted_agebs):
            peso = matriz[i, j]
            if peso > 0:
                G.add_edge(origin, dest, weight=peso)
    return G


def contar_superaciones(matriz_observada, lista_simulaciones):
    n_simulaciones = len(lista_simulaciones)
    shape = matriz_observada.shape
    conteo_superaciones = np.zeros(shape, dtype=int)
    for sim in lista_simulaciones:
        conteo_superaciones += (sim >= matriz_observada) 
    return conteo_superaciones, n_simulaciones


def calcular_significancia(conteo_superaciones, total_simulaciones, umbral=0.05):
    p_valores = (conteo_superaciones + 1) / (total_simulaciones + 1)
    significancia = p_valores <= umbral
    return significancia, p_valores


def obtener_matriz_significativa(matriz_significancia, agebs_nombres):
    n = matriz_significancia.shape[0]
    matriz_conteo = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            if i != j and matriz_significancia[i][j]:
                matriz_conteo[i, j] += 1
    return matriz_conteo


def calcular_inflow_outflow(matriz_flujo, agebs_nombres):
    n = matriz_flujo.shape[0]
    inflow_dict = {}
    outflow_dict = {}
    for i in range(n):
        outflow = np.sum(np.delete(matriz_flujo[i, :], i))
        inflow = np.sum(np.delete(matriz_flujo[:, i], i))
        nombre = agebs_nombres[i]
        inflow_dict[nombre] = inflow
        outflow_dict[nombre] = outflow
    return inflow_dict, outflow_dict


def comparar_simulaciones(simulaciones, observado, agebs_nombres):
    inflow_obs, outflow_obs = calcular_inflow_outflow(observado, agebs_nombres)
    n = len(agebs_nombres)
    total_simulaciones = len(simulaciones)
    conteo_superaciones = np.zeros((2, n), dtype=int)  
    for matriz_sim in simulaciones:
        inflow_sim, outflow_sim = calcular_inflow_outflow(matriz_sim, agebs_nombres)
        for idx, nombre in agebs_nombres.items():
            if inflow_sim[nombre] >= inflow_obs[nombre]: 
                conteo_superaciones[0, idx] += 1
            if outflow_sim[nombre] >= outflow_obs[nombre]: 
                conteo_superaciones[1, idx] += 1
    p_valores = (conteo_superaciones + 1) / (total_simulaciones + 1)
    significativos = p_valores <= 0.05
    return conteo_superaciones, significativos, p_valores


def colorear_nodos_por_significancia(G, significativos_inflow_outflow):
    inflow_sig, outflow_sig = significativos_inflow_outflow
    nodos = list(G.nodes)
    for idx, nodo in enumerate(nodos):
        inflow = inflow_sig[idx]
        outflow = outflow_sig[idx]
        if inflow and outflow:
            color = "orange"   # Thoroughfare
        elif inflow:
            color = "blue"     # Sink
        elif outflow:
            color = "red"      # Source
        else:
            color = "gray"     # Neutral
        G.nodes[nodo]['color'] = color
    return G


def graficar_grafo_coloreado_y_agebs(grafo, gdf_agebs, crimen, datos):
    fig, ax = plt.subplots(figsize=(10, 10))
    gdf_agebs.plot(ax=ax, color='none', edgecolor='gray', linewidth=0.5, alpha=0.7)
    colores = {'red': [], 'blue': [], 'orange': [], 'gray': []}
    for _, data in grafo.nodes(data=True):
        color = data.get('color', 'gray')
        colores[color].append((data['lon'], data['lat']))
    for color, coords in colores.items():
        if coords:
            x, y = zip(*coords)
            ax.scatter(x, y, color=color, s=10, label=color.capitalize())
    for i, (u, v) in enumerate(grafo.edges()):
        x_coords = [grafo.nodes[u]['lon'], grafo.nodes[v]['lon']]
        y_coords = [grafo.nodes[u]['lat'], grafo.nodes[v]['lat']]
        label = "Aristas" if i == 0 else None
        ax.plot(x_coords, y_coords, color='black', linewidth=0.5, alpha=0.6, label=label)
    leyenda_personalizada = [
        mpatches.Patch(color='red', label='Source (Significant outflow)'),
        mpatches.Patch(color='blue', label='Sink (Significant inflow)'),
        mpatches.Patch(color='orange', label='Thoroughfare (Both significant)'),
        mpatches.Patch(color='gray', label='Neutral (None significant)'),
        mpatches.Patch(color='black', label='Directed edges')
    ]
    ax.legend(handles=leyenda_personalizada, loc='upper left')
    ax.set_xlabel("Longitude (EPSG:32614)")
    ax.set_ylabel("Latitude (EPSG:32614)")
    #titulo = f"Criminal network of {crimen} (2017 - 2024) over Morelia AGEBs ({datos})"
    #ax.set_title(titulo)
    ax.set_title("")
    plt.tight_layout()
    nombre_archivo = f"red_{crimen.lower().replace(' ', '_')}.png"
    plt.savefig(nombre_archivo, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)
   

