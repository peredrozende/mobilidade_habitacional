import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from sklearn.base import BaseEstimator, ClusterMixin


class STDBSCAN(BaseEstimator, ClusterMixin):
    """Clusterização Espaço-Temporal ST-DBSCAN acelerada por KD-Tree.

    Atende ao padrão de interface da biblioteca sktime / scikit-learn.

    Parâmetros
    ----------
    eps1 : float, default=100.0
        Raio de tolerância espacial máximo (na unidade do CRS/coordenadas, ex: metros).
    eps2 : float, default=3600.0
        Janela de tolerância temporal máxima em segundos (ou mesma unidade da coluna de tempo).
    min_samples : int, default=3
        Número mínimo de pontos na vizinhança para formar um ponto central (core point).
    spatial_cols : tuple, default=('x', 'y')
        Nomes ou índices das colunas de coordenadas espaciais (usado em DataFrames).
    time_col : str, default='datetime'
        Nome da coluna com a data/hora ou timestamp numérico.
    """

    def __init__(
        self,
        eps1: float = 100.0,
        eps2: float = 3600.0,
        min_samples: int = 3,
        spatial_cols: tuple = ("x", "y"),
        time_col: str = "datetime",
    ):
        self.eps1 = eps1
        self.eps2 = eps2
        self.min_samples = min_samples
        self.spatial_cols = spatial_cols
        self.time_col = time_col

    def _extract_data(self, X):
        """Extrai a matriz espacial e o vetor temporal de GeoDataFrames, DataFrames ou NDArrays."""
        # Suporte a GeoDataFrame
        if hasattr(X, "geometry") and hasattr(X.geometry, "x"):
            spatial_data = np.column_stack([X.geometry.x.values, X.geometry.y.values])
            time_data = X[self.time_col].values
        # Suporte a DataFrame tradicional
        elif isinstance(X, pd.DataFrame):
            spatial_data = X[list(self.spatial_cols)].values
            time_data = X[self.time_col].values
        # Suporte a NumPy ndarray
        else:
            X_arr = np.asarray(X)
            spatial_data = X_arr[:, self.spatial_cols]
            time_data = X_arr[:, self.time_col]

        # Converte coluna datetime em segundos (epoch timestamp) se necessário
        if np.issubdtype(time_data.dtype, np.datetime64):
            time_data = time_data.astype("datetime64[s]").astype(np.int64)

        return np.asarray(spatial_data, dtype=np.float64), np.asarray(time_data, dtype=np.float64)

    def _get_st_neighbors(self, tree: KDTree, spatial_data: np.ndarray, time_data: np.ndarray, point_idx: int):
        """Busca vizinhos espaciais usando KD-Tree e aplica o filtro temporal."""
        # 1. Consulta KD-Tree em tempo O(log N) para candidatos no raio espacial eps1
        spatial_candidates = tree.query_ball_point(spatial_data[point_idx], r=self.eps1)

        # 2. Filtra candidatos pela distância temporal eps2
        st_neighbors = [
            j for j in spatial_candidates if abs(time_data[j] - time_data[point_idx]) <= self.eps2
        ]

        return st_neighbors

    def fit(self, X, y=None):
        """Executa a clusterização ST-DBSCAN sobre os dados informados."""
        spatial_data, time_data = self._extract_data(X)
        n_samples = len(spatial_data)

        # Construção da KD-Tree espacial
        tree = KDTree(spatial_data)

        labels = np.full(n_samples, -1, dtype=int)  # -1 indica ruído
        visited = np.zeros(n_samples, dtype=bool)
        cluster_id = 0

        for i in range(n_samples):
            if visited[i]:
                continue

            visited[i] = True
            neighbors = self._get_st_neighbors(tree, spatial_data, time_data, i)

            if len(neighbors) >= self.min_samples:
                labels[i] = cluster_id
                queue = set(neighbors)
                queue.discard(i)

                while queue:
                    q_idx = queue.pop()

                    if not visited[q_idx]:
                        visited[q_idx] = True
                        q_neighbors = self._get_st_neighbors(tree, spatial_data, time_data, q_idx)

                        if len(q_neighbors) >= self.min_samples:
                            queue.update(q_neighbors)
                            queue.discard(q_idx)

                    if labels[q_idx] == -1:
                        labels[q_idx] = cluster_id

                cluster_id += 1

        self.labels_ = labels
        return self

    def fit_predict(self, X, y=None):
        """Ajusta o modelo aos dados e retorna o vetor de rótulos dos clusters."""
        self.fit(X, y)
        return self.labels_


# --- Exemplo Completo de Uso com GeoDataFrame ---
if __name__ == "__main__":
    # Exemplo de dados de entrada
    raw_data = {
        "timestamp": pd.to_datetime([
            "2026-10-06 10:00:00",
            "2026-10-06 10:05:00",
            "2026-10-06 10:10:00",
            "2026-10-06 14:00:00",
            "2026-10-06 14:02:00"
        ]),
        "latitude": [-23.5505, -23.5506, -23.5507, -23.5900, -23.5901],
        "longitude": [-46.6333, -46.6334, -46.6335, -46.6800, -46.6801]
    }

    # Criando o GeoDataFrame e convertendo para sistema de projeção plana em metros (UTM)
    gdf = gpd.GeoDataFrame(
        raw_data,
        geometry=gpd.points_from_xy(raw_data["longitude"], raw_data["latitude"]),
        crs="EPSG:4326"
    ).to_crs(epsg=31983)

    # Instanciando o modelo
    st_dbscan = STDBSCAN(
        eps1=150.0,           # Raio espacial de 150 metros
        eps2=900.0,           # Raio temporal de 900 segundos (15 minutos)
        min_samples=2,        # Mínimo de 2 pontos para cluster
        time_col="timestamp"
    )

    # Executando a clusterização
    gdf["cluster"] = st_dbscan.fit_predict(gdf)

    print(gdf[["timestamp", "cluster"]])