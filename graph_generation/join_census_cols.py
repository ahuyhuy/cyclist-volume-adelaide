# POW_CTW: N people who work in this DZN and cycle to work
# POW_FE: N people who work in this DZN and are fully engaged in employment or study
# M_30_40, F_15_25: Age demographics - males 30-40 and females 15-25
# H_3500: Households with a THI of > 3500, place of enumeration, MB

import pandas as pd
import numpy as np
import networkx as nx


def _build_lookup(df, key_col, value_col):
    """Build dict key_col -> value_col, coercing key to str, value to numeric."""
    df = df.copy()
    df[key_col] = df[key_col].astype(str)
    df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
    return df.set_index(key_col)[value_col].to_dict()


def _map_nodes_to_values(node_codes, lookup):
    """Map node_id -> value using node_codes dict and lookup dict."""
    node_vals = {}
    for n, code in node_codes.items():
        code_str = str(code)
        node_vals[n] = lookup.get(code_str, np.nan)
    return node_vals


def _set_edge_attr_from_nodes(G, node_vals, attr_name, agg="avg_nonan"):
    """
    Set edge attribute from node values.
    
    agg: "avg_nonan" = average of u,v ignoring NaN; if both NaN -> NaN
         "max" = max of u,v
    """
    edge_dict = {}
    for u, v, k in G.edges(keys=True):
        u_val = node_vals.get(u, np.nan)
        v_val = node_vals.get(v, np.nan)
        
        if agg == "avg_nonan":
            vals = [x for x in (u_val, v_val) if not np.isnan(x)]
            edge_dict[(u, v, k)] = np.mean(vals) if vals else np.nan
        elif agg == "max":
            vals = [x for x in (u_val, v_val) if not np.isnan(x)]
            edge_dict[(u, v, k)] = max(vals) if vals else np.nan
        else:
            edge_dict[(u, v, k)] = np.nan
    
    nx.set_edge_attributes(G, edge_dict, attr_name)
    return G


def join_pow_ctw(
    G,
    csv_path="../all-data/census-formatted/greater-adelaide/MTW-Bicycle-POW-DZN.csv"
):
    """Add POW_CTW edge attribute: cyclists commuting to work in DZN."""
    try:
        df = pd.read_csv(csv_path)
        dzn_col = 'DZN (POW)' if 'DZN (POW)' in df.columns else df.columns[0]
        val_col = 'Bicycle' if 'Bicycle' in df.columns else df.columns[1]
        
        lookup = _build_lookup(df, dzn_col, val_col)
        
        node_dzn = nx.get_node_attributes(G, 'DZN_CODE21')
        if not node_dzn:
            print("Warning: No DZN_CODE21 node attributes for POW_CTW")
            return G
        
        node_vals = _map_nodes_to_values(node_dzn, lookup)
        G = _set_edge_attr_from_nodes(G, node_vals, 'POW_CTW', agg="avg_nonan")
        
        print(f"POW_CTW added: {len([v for v in node_vals.values() if not np.isnan(v)])} nodes matched")
        return G
    except Exception as e:
        print(f"Error in join_pow_ctw: {type(e).__name__}: {e}")
        return G


def join_pow_fe(
    G,
    csv_path="../all-data/census-formatted/greater-adelaide/EETP-POW-DZN.csv"
):
    """Add POW_FE edge attribute: fully engaged workers in DZN."""
    try:
        df = pd.read_csv(csv_path)
        dzn_col = 'DZN (POW)' if 'DZN (POW)' in df.columns else df.columns[0]
        val_col = 'Fully engaged' if 'Fully engaged' in df.columns else df.columns[1]
        
        lookup = _build_lookup(df, dzn_col, val_col)
        
        node_dzn = nx.get_node_attributes(G, 'DZN_CODE21')
        if not node_dzn:
            print("Warning: No DZN_CODE21 node attributes for POW_FE")
            return G
        
        node_vals = _map_nodes_to_values(node_dzn, lookup)
        G = _set_edge_attr_from_nodes(G, node_vals, 'POW_FE', agg="avg_nonan")
        
        print(f"POW_FE added: {len([v for v in node_vals.values() if not np.isnan(v)])} nodes matched")
        return G
    except Exception as e:
        print(f"Error in join_pow_fe: {type(e).__name__}: {e}")
        return G


def join_age_dems(
    G,
    csv_path="../all-data/census-formatted/greater-adelaide/A5YG-S-UR-MB.csv"
):
    """Add M_30_40 and F_15_25 edge attributes from MB-level age data."""
    try:
        df = pd.read_csv(csv_path)
        mb_col = 'MB (UR)' if 'MB (UR)' in df.columns else df.columns[0]
        
        # Column names from CSV inspection
        m30_34 = 'M 30-34 years'
        m35_39 = 'M 35-39 years'
        f15_19 = 'F 15-19 years'
        f20_24 = 'F 20-24 years'
        
        df = df.copy()
        df[mb_col] = df[mb_col].astype(str)
        
        # Coerce age columns to numeric
        for col in [m30_34, m35_39, f15_19, f20_24]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                print(f"Warning: Column '{col}' not found in age demographics CSV")
                df[col] = np.nan
        
        # Compute summed demographics
        df['M_30_40'] = df[m30_34].fillna(0) + df[m35_39].fillna(0)
        df['F_15_25'] = df[f15_19].fillna(0) + df[f20_24].fillna(0)
        
        # Build lookups
        m_lookup = df.set_index(mb_col)['M_30_40'].to_dict()
        f_lookup = df.set_index(mb_col)['F_15_25'].to_dict()
        
        node_mb = nx.get_node_attributes(G, 'MB_CODE21')
        if not node_mb:
            print("Warning: No MB_CODE21 node attributes for age demographics")
            return G
        
        # Map nodes
        m_node_vals = {n: m_lookup.get(str(mb), np.nan) for n, mb in node_mb.items()}
        f_node_vals = {n: f_lookup.get(str(mb), np.nan) for n, mb in node_mb.items()}
        
        # Set edge attributes (average of u,v, ignoring NaN)
        G = _set_edge_attr_from_nodes(G, m_node_vals, 'M_30_40', agg="avg_nonan")
        G = _set_edge_attr_from_nodes(G, f_node_vals, 'F_15_25', agg="avg_nonan")
        
        m_matched = len([v for v in m_node_vals.values() if not np.isnan(v)])
        f_matched = len([v for v in f_node_vals.values() if not np.isnan(v)])
        print(f"Age demographics added: M_30_40={m_matched} nodes, F_15_25={f_matched} nodes matched")
        return G
    except Exception as e:
        print(f"Error in join_age_dems: {type(e).__name__}: {e}")
        return G


def join_h_3500(
    G,
    csv_path="../all-data/census-formatted/greater-adelaide/THI-PE-MB.csv"
):
    """
    Add H_3500 edge attribute: households with weekly income >= $3500.
    
    Sums columns 17-22 (0-indexed) which correspond to:
    17: $3,500-$3,999
    18: $4,000-$4,499
    19: $4,500-$4,999
    20: $5,000-$5,999
    21: $6,000-$7,999
    22: $8,000 or more
    """
    try:
        df = pd.read_csv(csv_path)
        mb_col = 'MB (EN)' if 'MB (EN)' in df.columns else df.columns[0]
        
        # Columns 17-22 (0-indexed) = income brackets >= $3500/week
        target_cols = df.columns[17:23].tolist()  # 17,18,19,20,21,22
        
        df = df.copy()
        df[mb_col] = df[mb_col].astype(str)
        
        # Coerce target columns to numeric
        for col in target_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Sum households in brackets >= $3500
        df['H_3500'] = df[target_cols].sum(axis=1, skipna=True)
        
        # Build lookup
        lookup = df.set_index(mb_col)['H_3500'].to_dict()
        
        node_mb = nx.get_node_attributes(G, 'MB_CODE21')
        if not node_mb:
            print("Warning: No MB_CODE21 node attributes for H_3500")
            return G
        
        node_vals = {n: lookup.get(str(mb), np.nan) for n, mb in node_mb.items()}
        G = _set_edge_attr_from_nodes(G, node_vals, 'H_3500', agg="avg_nonan")
        
        matched = len([v for v in node_vals.values() if not np.isnan(v)])
        print(f"H_3500 added: {matched} nodes matched")
        return G
    except Exception as e:
        print(f"Error in join_h_3500: {type(e).__name__}: {e}")
        return G


def join_all(G):
    """Join all census columns into graph edges."""
    G = join_pow_ctw(G)
    G = join_pow_fe(G)
    G = join_age_dems(G)
    G = join_h_3500(G)
    return G