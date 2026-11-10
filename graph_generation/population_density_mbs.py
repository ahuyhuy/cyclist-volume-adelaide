"""
Population Density by Mesh Block (PD_MB)

Adds edge attribute PD_MB = max population density (persons/km²) of the 
edge's two endpoint nodes, based on their MB_CODE21.
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import networkx as nx
import os


def join_pd_mb(
    G,
    shp_path="../all-data/australia-mesh-blocks-raw/MB_2021_AUST_GDA2020.shp",
    csv_path="../all-data/census-formatted/greater-adelaide/P-UR-MB.csv",
    density_unit="per_km2"  # "per_km2" or "per_m2"
):
    """
    Add PD_MB edge attribute: max population density of edge's two nodes.
    
    Args:
        G: NetworkX graph with nodes having MB_CODE21 attribute
        shp_path: Path to MB shapefile with MB_CODE21, AREASQKM21
        csv_path: Path to population CSV with MB (UR), Total
        density_unit: "per_km2" (default) or "per_m2"
        
    Returns:
        Graph with PD_MB edge attribute, or original G on any error
    """
    try:
        # 1. Load MB shapefile with area (km²) and geometry
        mb_gdf = gpd.read_file(shp_path)[['MB_CODE21', 'AREASQKM21', 'geometry']]
        mb_gdf['MB_CODE21'] = mb_gdf['MB_CODE21'].astype(str)
        
        # 2. Load population CSV
        pop_df = pd.read_csv(csv_path)
        # Handle column names: 'MB (UR)', 'Total'
        mb_col = 'MB (UR)' if 'MB (UR)' in pop_df.columns else pop_df.columns[0]
        pop_col = 'Total' if 'Total' in pop_df.columns else pop_df.columns[1]
        pop_df = pop_df.rename(columns={mb_col: 'MB_CODE21', pop_col: 'population'})
        pop_df['MB_CODE21'] = pop_df['MB_CODE21'].astype(str)
        pop_df['population'] = pd.to_numeric(pop_df['population'], errors='coerce')
        
        # 3. Merge population into shapefile
        mb_gdf = mb_gdf.merge(pop_df, on='MB_CODE21', how='left')
        
        # 4. Compute population density (persons per km²)
        # Handle zero/NaN area
        mb_gdf['AREASQKM21'] = pd.to_numeric(mb_gdf['AREASQKM21'], errors='coerce')
        mb_gdf['pop_density_km2'] = np.where(
            (mb_gdf['AREASQKM21'] > 0) & (mb_gdf['population'].notna()),
            mb_gdf['population'] / mb_gdf['AREASQKM21'],
            0.0
        )
        
        # Convert to per m² if requested
        if density_unit == "per_m2":
            mb_gdf['pop_density'] = mb_gdf['pop_density_km2'] / 1_000_000
        else:
            mb_gdf['pop_density'] = mb_gdf['pop_density_km2']
        
        # 5. Create lookup dict: MB_CODE21 -> density
        density_lookup = mb_gdf.set_index('MB_CODE21')['pop_density'].to_dict()
        
        # 6. Get node MB_CODE21 attributes
        node_mb = nx.get_node_attributes(G, 'MB_CODE21')
        if not node_mb:
            print("Warning: No MB_CODE21 node attributes found")
            return G
        
        # 7. Map each node to its population density
        node_density = {}
        for n, mb in node_mb.items():
            mb_str = str(mb)
            node_density[n] = density_lookup.get(mb_str, 0.0)
        
        # 8. Compute edge PD_MB = max(u_density, v_density)
        pd_mb_dict = {}
        for u, v, k in G.edges(keys=True):
            u_dens = node_density.get(u, 0.0)
            v_dens = node_density.get(v, 0.0)
            pd_mb_dict[(u, v, k)] = max(u_dens, v_dens)
        
        # 9. Assign back to graph
        nx.set_edge_attributes(G, pd_mb_dict, 'PD_MB')
        
        # Quick stats
        vals = list(pd_mb_dict.values())
        if vals:
            print(f"PD_MB stats: min={min(vals):.2f}, max={max(vals):.2f}, mean={np.mean(vals):.2f} ({density_unit})")
            zero_count = sum(1 for v in vals if v == 0)
            print(f"  Edges with zero density: {zero_count}/{len(vals)}")
        
        return G
        
    except Exception as e:
        print(f"Error in join_pd_mb: {type(e).__name__}: {e}")
        print("Returning original graph unchanged")
        return G