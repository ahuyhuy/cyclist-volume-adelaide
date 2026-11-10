"""
Reusable module for fetching OSM bicycle route relations and adding them as edge attributes.
Works with overpy and uses reliable Overpass endpoints with retry logic.
"""

import time
import overpy
from typing import Set


# Reliable Overpass endpoint (kumi.systems works well)
#OVERPASS_ENDPOINT = "https://overpass.kumi.systems/api/interpreter"
#OVERPASS_ENDPOINT = "https://overpass-api.de/api/interpreter"
OVERPASS_ENDPOINT = "https://overpass.private.coffee/api/interpreter"


def get_cycle_route_way_ids(bbox: tuple, timeout: int = 180, max_retries: int = 2) -> Set[str]:
    """
    Get OSM way IDs for bicycle routes by splitting bbox into 4 sub-queries.
    More reliable for large areas (Adelaide metro).
    """
    import overpy
    import time
    from typing import Set
    
    minx, miny, maxx, maxy = bbox
    midx = (minx + maxx) / 2
    midy = (miny + maxy) / 2
    
    # 4 sub-bboxes: SW, SE, NW, NE
    sub_bboxes = [
        (minx, miny, midx, midy),   # SW
        (midx, miny, maxx, midy),   # SE
        (minx, midy, midx, maxy),   # NW
        (midx, midy, maxx, maxy),   # NE
    ]
    
    all_way_ids = set()
    api = overpy.Overpass(url=OVERPASS_ENDPOINT)
    
    for i, (sminx, sminy, smaxx, smaxy) in enumerate(sub_bboxes):
        query = f"""
        [out:json][timeout:{timeout}];
        (
          relation["route"="bicycle"]({sminy},{sminx},{smaxy},{smaxx});
        );
        way(r);
        out ids;
        """
        
        for attempt in range(max_retries):
            try:
                result = api.query(query)
                way_ids = {str(way.id) for way in result.ways}
                all_way_ids.update(way_ids)
                print(f"  Sub-query {i+1}/4: found {len(way_ids)} ways")
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    print(f"  Sub-query {i+1}/4 error: {type(e).__name__}: {e} — retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"  Sub-query {i+1}/4 failed after {max_retries} attempts: {type(e).__name__}: {e}")
                    # Continue to next sub-query instead of raising
    
    print(f"Total ways found across 4 sub-queries: {len(all_way_ids)}")
    return all_way_ids


def add_cycle_route_attribute(edges, cycle_way_ids: Set[str]):
    """
    Add IS_CYCLE_ROUTE column to edges GeoDataFrame.
    
    Args:
        edges: GeoDataFrame with 'osmid' column (from ox.graph_to_gdfs)
        cycle_way_ids: Set of way IDs from get_cycle_route_way_ids()
        
    Returns:
        Modified edges GeoDataFrame with IS_CYCLE_ROUTE column (0/1)
    """
    import pandas as pd
    
    def is_cycle_route(osmid):
        try:
            if pd.isna(osmid):
                return False
            elif hasattr(osmid, '__len__') and not isinstance(osmid, (str, bytes)):
                if len(osmid) == 0:
                    return False
                return any(str(osm_id) in cycle_way_ids for osm_id in osmid)
            else:
                return str(osmid) in cycle_way_ids
        except (TypeError, ValueError):
            return False
    
    edges = edges.copy()
    edges['IS_CYCLE_ROUTE'] = edges['osmid'].apply(is_cycle_route).astype(int)
    return edges


# Convenience function for common workflow
def enrich_graph_with_cycle_routes(G, bbox: tuple = None):
    """
    Add IS_CYCLE_ROUTE edge attribute to an OSMnx graph.
    
    Args:
        G: NetworkX graph from OSMnx
        bbox: Optional (minx, miny, maxx, maxy). If None, uses graph's bounding box.
        
    Returns:
        Graph with IS_CYCLE_ROUTE edge attribute
    """
    import osmnx as ox
    
    if bbox is None:
        nodes, edges = ox.graph_to_gdfs(G, nodes=True, edges=True)
        bbox = edges.total_bounds  # (minx, miny, maxx, maxy)
    
    cycle_way_ids = get_cycle_route_way_ids(bbox)
    print(f"Found {len(cycle_way_ids)} ways in bicycle routes")
    
    nodes, edges = ox.graph_to_gdfs(G, nodes=True, edges=True)
    edges = add_cycle_route_attribute(edges, cycle_way_ids)
    
    return ox.graph_from_gdfs(nodes, edges)