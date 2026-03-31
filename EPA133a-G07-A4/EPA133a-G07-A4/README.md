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



The network\_model.csv file is then used by the model file in the model folder. The model calculates and exports truck driving times times for each scenario and replication. These calculations are executed by running the model\_run file, which also stores the outputs in the Experiments folder.



Finally, there are 2 files important for the results visualisation . Plot bridge\_traffic\_by\_scenario plots  makes one network map per scenario plotting each bridge and their criticality on it.It also labels the top 5 busiest bridges in that scenario. Secondly, running plot\_experiment\_results which is visualising the more analysis oriented comparisons of the experiments, it uses the experiment outputs to generate visualizations, which are saved in the img folder.





## Requirements

To be able to run the model mesa version 2.1.4, pandas module 2.1.3 or later is needed and geopandas for the compare\_intersections\_with\_shapefile.py

