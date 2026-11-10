# Modelling Cyclist Volumes In Adelaide, South Australia

Alexander Huynh, Adelaide University.

## Introduction

This project aims to lay a basic foundation for Cyclist Volume Modelling in Adelaide. For the full report see `REPORT (replace later)`. The product of this project is a predictive model for the Greater Adelaide region, viewable by opening `model_docs/model_data/prediction/model_4_interactive.qgz` in QGIS. A simplified preview of the model results is shown below.

![](model_docs/model_data/prediction/predicted_edges.png)

The model was created using segmented linear regression with ordinary least squares (OLS). `model_docs/model_notebook.ipynb` contains the regression process and `model_docs/model_data/` contains the design matrices and regression results. This work is designed to be continued, and the regression model is also refreshable - the __Model Refresh Tool__ allows for more up-to-date or comprehensive data to be easily used to recalibrate the model's coefficients.

## Model Refresh Tool

`script here`

https://drive.google.com/drive/folders/1tO3klCw9SWAbWICeORm5qa-w-bNdsnZu?usp=sharing

## Handover Notes







## Directory guide

`all-data/`: Contains census data, geo data for census data geographical regions (mesh blocks, destination zones), and cyclist count data

`bicycling-infrastructure-classification/`: A different repository for __Developing a bicycling infrastructure classification system for Greater Melbourne, Australia using OpenStreetMap__ by Debjit Bhowmick and Ben Beck. This work formed the basis of the bicycle infrastructure classification for this model. https://www.researchgate.net/publication/373296593_Developing_a_bicycling_infrastructure_classification_system_for_Greater_Melbourne_Australia_using_OpenStreetMap

`graph_generation/`: Python notebooks for the Greater Adelaide cycle network graph setup and graph loading. These notebooks combined create the graph network that can be used by any regression model.

`investigation/`: Earlier notebooks, test scripts, and models

`scripts/`: Shared code used in graph generation

# AI Usage (move into final report)

__NVIDIA Nemotron 3 Ultra (free)__ was the AI model used to assist with coding in python and managing the python environment. It was run __Deepseek Harness__ (see https://www.deepseek.com/en/harness/) - a locally run code harness that enables file read/write and maintains work context across sessions and models. Deepseek Harness was connected to the chosen model via __OpenRouter__, a service allowing users to connect harnesses such as copilot, claude code, cursor, and deepseek harness, to any available LLM model. This setup was primarily used for coding and technical support, not for design decisions or documentation.