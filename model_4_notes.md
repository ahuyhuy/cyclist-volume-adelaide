# Segment variables

Decisions have been made on what vars to drop based on column holdout validation - invididual columns were withheld from different segments based on correlation and coefficient values.

## CBD

- POW_CTW: N people who work in this DZN and cycle to work
- POW_FE: N people who work in this DZN and are fully engaged in employment or study
- PD_MB: MB population density
- DNL_DZN: Distance to nearest large DZN
- M_30_40, F_15_25: Age demographics, males 30-40 and females 15-25
- INF_CR: Listed cycle routes
- MBI_PTL: Painted bike lane, Painted bike lane (single-side)
- INF_LgRd: OSM 'highway:primary' or 'highway:secondary'
- INF_SmRd: OSM 'highway:tertiary' or 'highway:residential'

__Dropped__

- MBI_SZ: Shared zone // NA
- MBI_Bus: Bus lane with cycling allowed // NA
- MBI_Mx: Mixed traffic // NA
- H_3500: Households with a THI of > 3500, place of enumeration, MB // CORRELATED (less relevant for CBD cos most cycle traffic originates from outside of CBD)
- MBI_BP: Dedicated bike path, Shared bike path // CORRELATED
- INF_TrRd: OSM 'highway:trunk' // NA

## CBDP

- POW_CTW: N people who work in this DZN and cycle to work
- POW_FE: N people who work in this DZN and are fully engaged in employment or study
- DNL_DZN: Distance to nearest large DZN
- INF_CR: Listed cycle routes
- MBI_BP: Dedicated bike path, Shared bike path
- MBI_PTL: Painted bike lane, Painted bike lane (single-side)
- MBI_SZ: Shared zone
- MBI_Bus: Bus lane with cycling allowed
- INF_SmRd: OSM 'highway:tertiary' or 'highway:residential' 

__Dropped__

- MBI_PP: Pedestrian path/street with cycling allowed // NA
- MBI_Sh: Sharrow // NA
- MBI_Mx: Mixed traffic // NA
- All MB related cols cos empty
- INF_LgRd: OSM 'highway:primary' or 'highway:secondary' // CORRELATED (all main roads in parklands have parallel bike paths so MBI_BP has it covered)
- INF_TrRd: OSM 'highway:trunk' // NA

## S

- POW_CTW: N people who work in this DZN and cycle to work
- POW_FE: N people who work in this DZN and are fully engaged in employment or study
- PD_MB: MB population density
- DNL_DZN: Distance to nearest large DZN
- M_30_40, F_15_25: Age demographics, males 30-40 and females 15-25
- H_3500: Households with a THI of > 3500, place of enumeration, MB
- INF_CR: Listed cycle routes
- INF_TrRd: OSM 'highway:trunk'
- INF_LgRd: OSM 'highway:primary' or 'highway:secondary'
- INF_SmRd: OSM 'highway:tertiary' or 'highway:residential'


__Dropped__

- MBI_BP: Dedicated bike path, Shared bike path // CORRELATED
- MBI_Sh: Sharrow // BAD
- MBI_PTL: Painted bike lane, Painted bike lane (single-side) // CORRELATED AND BAD
- MBI_PP: Pedestrian path/street with cycling allowed // NA
- MBI_SZ: Shared zone // NA
- MBI_Bus: Bus lane with cycling allowed // NA
- MBI_Mx: Mixed traffic // NA

## SP

- POW_CTW: N people who work in this DZN and cycle to work
- POW_FE: N people who work in this DZN and are fully engaged in employment or study
- DNL_DZN: Distance to nearest large DZN
- INF_CR: Listed cycle routes
- MBI_PP: Pedestrian path/street with cycling allowed
- INF_TrRd: OSM 'highway:trunk'
- INF_LgRd: OSM 'highway:primary' or 'highway:secondary'
- MBI_BP: Dedicated bike path, Shared bike path

__Dropped__

- MBI_Sh: Sharrow // NA
- MBI_Bus: Bus lane with cycling allowed // NA
- MBI_Mx: Mixed traffic // NA
- MBI_SZ: Shared zone // NA
- MBI_PTL: Painted bike lane, Painted bike lane (single-side) // BAD, CORRELATED
- INF_SmRd: OSM 'highway:tertiary' or 'highway:residential' // BAD, CORRELATED

