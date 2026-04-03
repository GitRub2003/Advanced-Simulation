# Assignment 3: Network model generation

Created by:

|Group Number|07|
|:-:|:-:|
|Ruben Zuidgeest|5542014|
|Lars Groen|4667697|
|Ryan Zondag|5543485|
|Olivier Poelman|5589177|
|Sybe de Haan|5595088|

## How to use

## 

In this file there are several important folders.



The first is the data folder. It contains the \_roads3 and BMMS\_overview CSV files, which are needed to produce the road network (network.csv) for the analysis.



The road network is produced by running 'run\_pipeline' , which runs 'Data Preparation multiple roads', 'intersection creation', 'Intersections in the csv' 'N106 check', road\_selection' 'roadmap plotting with bridges' and 'side roads choosing'. More details on this are provided in the report.



The network\_model.csv file is then used by the model file in the model folder. The model calculates and exports infrastructure crossings and truck driving times times for each scenario and replication. These calculations are executed by running the model\_run file, which also stores the outputs in the Experiments folder.



Finally, there are 3 files important for the results visualisation . First run `plot\_experiment\_results.py`, which reads the raw scenario and replicate CSV files from `Experiments/` and writes summary CSVs and plots to `img/`. Then run `plot\_bridge\_traffic\_by\_scenario.py`, which uses `data/network\_model.csv` and `img/infrastructure\_importance\_by\_scenario.csv` to create bridge traffic maps for each scenario. `plot\_top\_bridges\_on\_map.py`highlights the bridges that are consistently among the busiest across scenarios. It also uses 'img/infrastructure\_importance\_by\_scenario.csv' and outputs plots in 'img'.



## Requirements

To be able to run the model mesa version 2.1.4, pandas module 2.1.3 or later is needed and geopandas for the compare\_intersections\_with\_shapefile.py. The latest pyDOE version is needed to run the latin hypercube experimental design in model\_run

