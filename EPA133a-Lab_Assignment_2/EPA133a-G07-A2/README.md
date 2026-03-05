# Assignment 2: Component Building

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

In this file there are several important folders. The first folder which is important is the data folder. This folder contains the \_roads3 and BMMS\_overview csv files which  are needed to produce driving and waiting times of the trucks (per bridge and road), more about this is written in the report. To be able to use this data, the Data Preparation file needs to be ran in pycharm to get this data in the right format, this file produces a csv file (n1\_model) with all the needed information in this format.



With this csv file the model file in the model folder generates a model which can calculate and export truck driving times and bridge total wait times per scenario and replications. These files are stored in the Experiments folder.

Finally, when running the stats\_components file in the report folder you create visualation which are then stored in the img folder

## Requirements

To be able to run the model mesa version 2.1.4 and pandas module 2.1.3 or later is needed. 

### 

