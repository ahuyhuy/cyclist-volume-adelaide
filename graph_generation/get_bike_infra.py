# Monash Bicycling Infrastructure Classifiers:
# • Dedicated bike path
# • Shared bike path
# • Pedestrian path/street with cycling allowed
# • Protected bike lane
# • Buffered bike lane (kerb-side)
# • Buffered bike lane (road-side)
# • Buffered bike lane (both sides)
# • Painted bike lane
# • Painted bike lane (single-side)
# • Advisory bike lane
# • Advisory bike lane (single-side)
# • Peak hour painted bike lane
# • Peak hour painted bike lane (single-side)
# • Peak hour advisory bike lane
# • Peak hour advisory bike lane (single-side)
# • Sharrow
# • Shared zone
# • Bus lane with cycling allowed
# • Mixed traffic
# • Bicyclists dismount
# • Pedestrian path/street with cycling not allowed

# These classifiers are aggregated into the following dummy variables, which should be loaded into the Graph

# - MBI_BP: Dedicated bike path, Shared bike path
# - MBI_PP: Pedestrian path/street with cycling allowed
# - MBI_PL: Protected bike lane
# - MBI_BL: Buffered bike lane (kerb-side), Buffered bike lane (road-side), Buffered bike lane (both sides)
# - MBI_PTL: Painted bike lane, Painted bike lane (single-side)
# - MBI_Ad: Advisory bike lane, Advisory bike lane (single-side), Peak hour advisory bike lane, Peak hour advisory bike lane (single-side)
# - MBI_PkPt: Peak hour painted bike lane, Peak hour painted bike lane (single-side)
# - MBI_Sh: Sharrow
# - MBI_SZ: Shared zone
# - MBI_Bus: Bus lane with cycling allowed
# - MBI_Mx: Mixed traffic
# - MBI_No: Bicyclists dismount, Pedestrian path/street with cycling not allowed

import osmnx as ox
import pandas as pd
import numpy as np


def _classify_smsr(edges: pd.DataFrame) -> pd.Series:
    """
    Apply Monash SMSR classification to edges DataFrame.
    
    Uses the Greater Melbourne variant (peak hour checks cycleway:left:conditional
    and cycleway:left:lane:conditional).
    """
    # Start with all mixed traffic
    bike_inf_smsr = pd.Series('1. mixed traffic', index=edges.index, dtype=object)
    
    # Helper: mask for "still mixed traffic"
    def mt_mask():
        return bike_inf_smsr == '1. mixed traffic'
    
    def mt_or_painted_mask():
        return bike_inf_smsr.isin(['1. mixed traffic', '5e. painted bikelane'])
    
    def mt_or_painted_single_mask():
        return bike_inf_smsr.isin(['5d. painted bikelane (single side)'])
    
    def advisory_mask():
        return bike_inf_smsr.isin(['5c. advisory bikelane'])
    
    def advisory_single_mask():
        return bike_inf_smsr.isin(['5b. advisory bikelane (single side)'])
    
    # Safe column access (columns may not exist if OSM didn't return them)
    def has_col(col):
        return col in edges.columns
    
    def col_eq(col, vals):
        if not has_col(col):
            return pd.Series(False, index=edges.index)
        return edges[col].isin(vals) if isinstance(vals, list) else (edges[col] == vals)
    
    def col_not_eq(col, vals):
        if not has_col(col):
            return pd.Series(True, index=edges.index)
        return ~edges[col].isin(vals) if isinstance(vals, list) else (edges[col] != vals)
    
    def col_notna(col):
        if not has_col(col):
            return pd.Series(False, index=edges.index)
        return edges[col].notna()
    
    # BICYCLISTS DISMOUNT
    bike_inf_smsr.loc[col_eq('bicycle', 'dismount') & mt_mask()] = '0a. bicyclists dismount'
    
    # BIKEPATH (dedicated)
    bike_inf_smsr.loc[col_eq('highway', 'cycleway') & col_eq('foot', 'no') & mt_mask()] = '9. dedicated bikepath'
    bike_inf_smsr.loc[col_eq('highway', 'path') & col_eq('bicycle', ['yes', 'designated']) & col_eq('foot', 'no') & mt_mask()] = '9. dedicated bikepath'
    
    # SHARED PATH
    bike_inf_smsr.loc[col_eq('highway', 'cycleway') & col_not_eq('foot', 'no') & mt_mask()] = '8b. shared bikepath'
    bike_inf_smsr.loc[col_eq('highway', 'path') & col_eq('bicycle', ['yes', 'designated']) & col_not_eq('foot', 'no') & mt_mask()] = '8b. shared bikepath'
    
    # PEDESTRIAN PATH WITH CYCLING ALLOWED
    bike_inf_smsr.loc[col_eq('highway', ['footway', 'pedestrian']) & col_eq('bicycle', ['yes', 'designated']) & mt_mask()] = '8a. pedestrian path/street with cycling allowed'
    
    # PEDESTRIAN PATH WITH CYCLING NOT ALLOWED
    bike_inf_smsr.loc[col_eq('highway', ['footway', 'pedestrian', 'path']) & col_not_eq('bicycle', ['yes', 'designated']) & mt_mask()] = '0b. pedestrian path/street with cycling not allowed'
    
    # PROTECTED BIKELANE
    protected_cond = (
        col_not_eq('highway', 'cycleway') & (
            col_eq('cycleway', ['track', 'opposite_track']) |
            col_eq('cycleway:left', 'track') |
            col_eq('cycleway:right', 'track') |
            col_eq('cycleway:both', 'track')
        )
    )
    bike_inf_smsr.loc[protected_cond & mt_mask()] = '7. protected bikelane'
    # St Kilda Rd style: cycleway tagged as highway=cycleway on a named road
    bike_inf_smsr.loc[col_eq('highway', 'cycleway') & edges['name'].astype(str).str.contains("Road|Street", na=False) & mt_mask()] = '7. protected bikelane'
    
    # BUFFERED BIKELANE (BOTH SIDES)
    buf_both_cond = (
        col_eq('cycleway:both', 'lane') &
        (col_eq('cycleway:both:lane', 'exclusive') | col_eq('cycleway:left:lane', 'exclusive')) &
        (
            col_notna('cycleway:left:buffer:both') | col_notna('cycleway:both:buffer:both') |
            (col_notna('cycleway:left:buffer:left') & col_notna('cycleway:left:buffer:right')) |
            (col_notna('cycleway:both:buffer:left') & col_notna('cycleway:both:buffer:right'))
        )
    )
    bike_inf_smsr.loc[buf_both_cond & mt_mask()] = '6c. buffered lane (both sides)'
    
    buf_both_left_cond = (
        col_eq('cycleway:left', 'lane') &
        col_eq('cycleway:left:lane', 'exclusive') &
        (
            (col_notna('cycleway:left:buffer:left') & col_notna('cycleway:left:buffer:right')) |
            col_notna('cycleway:left:buffer:both')
        )
    )
    bike_inf_smsr.loc[buf_both_left_cond & mt_mask()] = '6c. buffered lane (both sides)'
    
    # BUFFERED BIKELANE (KERB-SIDE)
    buf_kerb_cond = (
        col_eq('cycleway:both', 'lane') &
        (col_eq('cycleway:both:lane', 'exclusive') | col_eq('cycleway:left:lane', 'exclusive')) &
        (col_notna('cycleway:left:buffer:left') | col_notna('cycleway:both:buffer:left')) &
        ~(col_notna('cycleway:left:buffer:right') | col_notna('cycleway:both:buffer:right'))
    )
    bike_inf_smsr.loc[buf_kerb_cond & mt_mask()] = '6a. buffered bikelane (kerb-side)'
    
    buf_kerb_left_cond = (
        col_eq('cycleway:left', 'lane') &
        col_eq('cycleway:left:lane', 'exclusive') &
        col_notna('cycleway:left:buffer:left') &
        col_eq('cycleway:left:buffer:right', pd.NA)  # explicitly NA
    )
    bike_inf_smsr.loc[buf_kerb_left_cond & mt_mask()] = '6a. buffered bikelane (kerb-side)'
    
    # BUFFERED BIKELANE (ROAD-SIDE)
    buf_road_cond = (
        col_eq('cycleway:both', 'lane') &
        (col_eq('cycleway:both:lane', 'exclusive') | col_eq('cycleway:left:lane', 'exclusive')) &
        (col_notna('cycleway:left:buffer:right') | col_notna('cycleway:both:buffer:right')) &
        ~(col_notna('cycleway:left:buffer:left') | col_notna('cycleway:both:buffer:left'))
    )
    bike_inf_smsr.loc[buf_road_cond & mt_mask()] = '6b. buffered bikelane (road-side)'
    
    buf_road_left_cond = (
        col_eq('cycleway:left', 'lane') &
        col_eq('cycleway:left:lane', 'exclusive') &
        col_notna('cycleway:left:buffer:right') &
        col_eq('cycleway:left:buffer:left', pd.NA)
    )
    bike_inf_smsr.loc[buf_road_left_cond & mt_mask()] = '6b. buffered bikelane (road-side)'
    
    # PAINTED BIKELANE
    painted_cond = (
        col_not_eq('highway', 'cycleway') & (
            col_eq('cycleway', ['lane', 'opposite_lane']) |
            col_eq('cycleway:left', 'lane') |
            col_eq('cycleway:right', 'lane') |
            col_eq('cycleway:both', 'lane')
        )
    )
    bike_inf_smsr.loc[painted_cond & mt_mask()] = '5e. painted bikelane'
    
    # PAINTED BIKELANE (SINGLE SIDE)
    painted_single_cond = (
        ((col_eq('cycleway:left', 'lane') & col_eq('cycleway:right', 'no')) |
         (col_eq('cycleway:right', 'lane') & col_eq('cycleway:left', 'no')))
    )
    bike_inf_smsr.loc[painted_single_cond & (bike_inf_smsr == '5e. painted bikelane')] = '5d. painted bikelane (single side)'
    
    # ADVISORY BIKELANE
    advisory_cond = (
        col_eq('cycleway:both', 'lane') &
        (col_eq('cycleway:both:lane', 'advisory') | col_eq('cycleway:left:lane', 'advisory'))
    )
    bike_inf_smsr.loc[advisory_cond & (bike_inf_smsr.isin(['5e. painted bikelane', '5d. painted bikelane (single side)']))] = '5c. advisory bikelane'
    
    advisory_left_cond = (
        col_eq('cycleway:left', 'lane') &
        col_eq('cycleway:left:lane', 'advisory')
    )
    bike_inf_smsr.loc[advisory_left_cond & (bike_inf_smsr.isin(['5e. painted bikelane', '5d. painted bikelane (single side)']))] = '5c. advisory bikelane'
    
    # ADVISORY BIKELANE (SINGLE SIDE)
    bike_inf_smsr.loc[
        col_eq('cycleway:left', 'lane') & col_eq('cycleway:right', 'no') & col_eq('cycleway:left:lane', 'advisory') & (bike_inf_smsr == '5c. advisory bikelane')
    ] = '5b. advisory bikelane (single side)'
    
    # PEAK HOUR BIKELANE (Greater Melbourne variant: checks cycleway:left:conditional and cycleway:left:lane:conditional)
    peak_cond = col_notna('cycleway:left:conditional') | col_notna('cycleway:left:lane:conditional')
    
    bike_inf_smsr.loc[peak_cond & mt_or_painted_mask()] = '5a. peak hour painted bikelane'
    bike_inf_smsr.loc[peak_cond & mt_or_painted_single_mask()] = '5a. peak hour painted bikelane (single side)'
    bike_inf_smsr.loc[peak_cond & advisory_mask()] = '5a. peak hour advisory bikelane'
    bike_inf_smsr.loc[peak_cond & advisory_single_mask()] = '5a. peak hour advisory bikelane (single side)'
    
    # SHARROW
    sharrow_cond = (
        (col_not_eq('highway', 'cycleway') & col_eq('cycleway', 'shared_lane')) |
        (col_eq('cycleway:both', 'shared_lane') & (col_eq('cycleway:both:lane', 'pictogram') | col_eq('cycleway:left:lane', 'pictogram'))) |
        (col_eq('cycleway:left', 'shared_lane') & col_eq('cycleway:left:lane', 'pictogram')) |
        (col_eq('cycleway:right', 'shared_lane') & col_eq('cycleway:left', 'no'))
    )
    bike_inf_smsr.loc[sharrow_cond & mt_mask()] = '4b. sharrow'
    
    # SHARED ZONE
    bike_inf_smsr.loc[col_eq('highway', 'living_street') & mt_mask()] = '4a. shared zone'
    
    # SHARED BUS LANE
    bus_cond = (
        col_eq('cycleway:both', ['share_busway', 'opposite_share_busway']) |
        col_eq('cycleway:left', ['share_busway', 'opposite_share_busway'])
    )
    bike_inf_smsr.loc[bus_cond & mt_mask()] = '3. bus lane with cycling allowed'
    
    # SHOULDER CYCLABLE
    shoulder_cond = col_eq('cycleway:both', 'shoulder') | col_eq('cycleway:left', 'shoulder')
    bike_inf_smsr.loc[shoulder_cond & mt_mask()] = '2. shoulder cyclable'
    
    # ASSOCIATED BIKE INFRASTRUCTURE SEPARATELY MAPPED
    bike_inf_smsr.loc[col_eq('cycleway', 'separate') & mt_mask()] = '0b. Associated bike infrastructure separately mapped'
    
    return bike_inf_smsr


def _add_mbi_dummies(edges: pd.DataFrame) -> pd.DataFrame:
    """
    Map SMSR classification to 12 MBI_ dummy variables.
    """
    edges = edges.copy()
    
    # Initialize all dummies to 0
    mbi_cols = [
        'MBI_BP', 'MBI_PP', 'MBI_PL', 'MBI_BL', 'MBI_PTL',
        'MBI_Ad', 'MBI_PkPt', 'MBI_Sh', 'MBI_SZ', 'MBI_Bus',
        'MBI_Mx', 'MBI_No'
    ]
    for col in mbi_cols:
        edges[col] = 0
    
    sms = edges['bike_inf_smsr']
    
    # MBI_BP: Dedicated bike path, Shared bike path
    edges.loc[sms.isin(['9. dedicated bikepath', '8b. shared bikepath']), 'MBI_BP'] = 1
    
    # MBI_PP: Pedestrian path/street with cycling allowed
    edges.loc[sms == '8a. pedestrian path/street with cycling allowed', 'MBI_PP'] = 1
    
    # MBI_PL: Protected bike lane
    edges.loc[sms == '7. protected bikelane', 'MBI_PL'] = 1
    
    # MBI_BL: Buffered bike lane (all variants)
    edges.loc[sms.isin([
        '6a. buffered bikelane (kerb-side)',
        '6b. buffered bikelane (road-side)',
        '6c. buffered lane (both sides)'
    ]), 'MBI_BL'] = 1
    
    # MBI_PTL: Painted bike lane (both variants)
    edges.loc[sms.isin(['5e. painted bikelane', '5d. painted bikelane (single side)']), 'MBI_PTL'] = 1
    
    # MBI_Ad: Advisory bike lane (including peak hour advisory)
    edges.loc[sms.isin([
        '5c. advisory bikelane',
        '5b. advisory bikelane (single side)',
        '5a. peak hour advisory bikelane',
        '5a. peak hour advisory bikelane (single side)'
    ]), 'MBI_Ad'] = 1
    
    # MBI_PkPt: Peak hour painted bike lane (both variants)
    edges.loc[sms.isin([
        '5a. peak hour painted bikelane',
        '5a. peak hour painted bikelane (single side)'
    ]), 'MBI_PkPt'] = 1
    
    # MBI_Sh: Sharrow
    edges.loc[sms == '4b. sharrow', 'MBI_Sh'] = 1
    
    # MBI_SZ: Shared zone
    edges.loc[sms == '4a. shared zone', 'MBI_SZ'] = 1
    
    # MBI_Bus: Bus lane with cycling allowed
    edges.loc[sms == '3. bus lane with cycling allowed', 'MBI_Bus'] = 1
    
    # MBI_Mx: Mixed traffic + shoulder cyclable
    edges.loc[sms.isin(['1. mixed traffic', '2. shoulder cyclable']), 'MBI_Mx'] = 1
    
    # MBI_No: Bicyclists dismount, Pedestrian path/street with cycling not allowed
    edges.loc[sms.isin([
        '0a. bicyclists dismount',
        '0b. pedestrian path/street with cycling not allowed'
    ]), 'MBI_No'] = 1
    
    # Note: '0b. Associated bike infrastructure separately mapped' gets no dummy (remains all 0)
    
    return edges


def enrich_graph_with_monash_classifiers(G):
    """
    Add Monash SMSR bicycle infrastructure classification and MBI_ dummy variables
    to an OSMnx graph.
    
    Args:
        G: NetworkX graph from OSMnx (must have OSM tags on edges)
        
    Returns:
        Graph with MBI_* edge attributes (12 dummy variables)
    """
    # Convert to nodes/edges GeoDataFrames
    nodes, edges = ox.graph_to_gdfs(G, nodes=True, edges=True)
    
    # Apply SMSR classification
    edges['bike_inf_smsr'] = _classify_smsr(edges)
    
    # Map to MBI dummy variables
    edges = _add_mbi_dummies(edges)
    
    # Rebuild graph with enriched edges
    G_enriched = ox.graph_from_gdfs(nodes, edges)
    
    return G_enriched