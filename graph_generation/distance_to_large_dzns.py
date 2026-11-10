"""
Distance to Nearest Large Destination Zone (DNL_DZN)

Adds edge attribute DNL_DZN = distance (meters) from edge midpoint to nearest
node belonging to a large DZN (>= threshold workers, POW). If edge's u or v
is a large-DZN node, distance = 0.
"""

import pandas as pd
import numpy as np
import osmnx as ox
import networkx as nx
from scipy.spatial import KDTree


def join_dnl_dzn(
    G,
    csv_path="../all-data/census-formatted/greater-adelaide/P-POW-DZN.csv",
    threshold=8000,
    target_crs="epsg:32754"  # UTM zone 54S for Adelaide
):
    """
    Add DNL_DZN edge attribute: distance (meters) to nearest large DZN node.
    
    Args:
        G: NetworkX graph with nodes having DZN_CODE21 attribute
        csv_path: Path to P-POW-DZN.csv with columns ['DZN (POW)', 'Total']
        threshold: Worker count threshold for "large" DZN (default 8000)
        target_crs: Metric CRS for distance calculation (default EPSG:32754)
        
    Returns:
        Graph with DNL_DZN edge attribute, or original G on any error
    """
    try:
        # 1. Load large DZN codes from CSV
        df = pd.read_csv(csv_path)
        # Handle column name variations
        dzn_col = 'DZN (POW)' if 'DZN (POW)' in df.columns else df.columns[0]
        total_col = 'Total' if 'Total' in df.columns else df.columns[1]
        
        # Coerce Total to numeric (handles commas, strings, etc.)
        df[total_col] = pd.to_numeric(df[total_col], errors='coerce')
        
        large_dzn_codes = set(
            df.loc[df[total_col] >= threshold, dzn_col].astype(str)
        )
        
        if not large_dzn_codes:
            print("Warning: No large DZNs found, setting DNL_DZN = -1")
            for u, v, k in G.edges(keys=True):
                G[u][v][k]['DNL_DZN'] = -1.0
            return G
        
        print(f"Found {len(large_dzn_codes)} large DZNs (>= {threshold} workers)")
        
        # 2. Get node DZN_CODE21 attributes and identify large-DZN nodes
        node_dzn = nx.get_node_attributes(G, 'DZN_CODE21')
        if not node_dzn:
            print("Warning: No DZN_CODE21 node attributes found")
            return G
        
        # Convert to Series for vectorized operations
        node_dzn_series = pd.Series(node_dzn)
        node_dzn_series.index = node_dzn_series.index.astype(int)
        
        # Flag large-DZN nodes
        is_large_dzn = node_dzn_series.astype(str).isin(large_dzn_codes)
        large_dzn_nodes = is_large_dzn[is_large_dzn].index.tolist()
        
        if not large_dzn_nodes:
            print("Warning: No graph nodes match large DZN codes")
            for u, v, k in G.edges(keys=True):
                G[u][v][k]['DNL_DZN'] = -1.0
            return G
        
        print(f"Found {len(large_dzn_nodes)} nodes in large DZNs")
        
        # 3. Get nodes GeoDataFrame and project to metric CRS
        nodes_gdf = ox.graph_to_gdfs(G, nodes=True, edges=False)[['geometry']]
        nodes_gdf = nodes_gdf.to_crs(target_crs)
        
        # Extract coordinates of large-DZN nodes for KDTree
        large_nodes_geom = nodes_gdf.loc[large_dzn_nodes]
        large_coords = np.column_stack([
            large_nodes_geom.geometry.x.values,
            large_nodes_geom.geometry.y.values
        ])
        
        # Build KDTree for fast nearest-neighbor search
        tree = KDTree(large_coords)
        
        # 4. Get edges with geometry and project
        edges_gdf = ox.graph_to_gdfs(G, nodes=False, edges=True)
        edges_gdf = edges_gdf.to_crs(target_crs)
        
        # Ensure we have edge geometry for midpoint calculation
        if 'geometry' not in edges_gdf.columns:
            print("Warning: No edge geometry found")
            return G
        
        # 5. Compute DNL_DZN for each edge
        dnl_dict = {}
        
        # Pre-compute set of large-DZN nodes for O(1) lookup
        large_dzn_set = set(large_dzn_nodes)
        
        for (u, v, k), row in edges_gdf.iterrows():
            # If either endpoint is a large-DZN node, distance = 0
            if u in large_dzn_set or v in large_dzn_set:
                dnl_dict[(u, v, k)] = 0.0
                continue
            
            # Compute edge midpoint
            geom = row['geometry']
            if geom is None or geom.is_empty:
                dnl_dict[(u, v, k)] = -1.0
                continue
            
            midpoint = geom.interpolate(0.5, normalized=True)
            mid_coord = np.array([[midpoint.x, midpoint.y]])
            
            # Query KDTree for nearest large-DZN node
            try:
                dist, _ = tree.query(mid_coord, k=1)
                dnl_dict[(u, v, k)] = float(dist[0])
            except Exception:
                dnl_dict[(u, v, k)] = -1.0
        
        # 6. Assign back to graph
        nx.set_edge_attributes(G, dnl_dict, 'DNL_DZN')
        
        # Quick stats
        vals = [v for v in dnl_dict.values() if v >= 0]
        if vals:
            print(f"DNL_DZN stats: min={min(vals):.0f}m, max={max(vals):.0f}m, mean={np.mean(vals):.0f}m")
            zero_count = sum(1 for v in vals if v == 0)
            print(f"  Edges in large DZNs (dist=0): {zero_count}/{len(vals)}")
        
        return G
        
    except Exception as e:
        print(f"Error in join_dnl_dzn: {type(e).__name__}: {e}")
        print("Returning original graph unchanged")
        return G