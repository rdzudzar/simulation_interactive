#!/usr/bin/env python

#Original code from swin cookies-n-code; now updated to work on DarkSage output
#To run it with csv:
#bokeh serve --allow-websocket-origin=localhost:3112 --port=3112 --port=3112 GUI_cookiesncode.py --args -csv computers.csv

#To run it directly with DarkSage output
#bokeh serve --allow-websocket-origin=localhost:3112 --port=3112 GUI_DarkSage.py


#General imports
import pandas as pd
import numpy as np
import os
import sys
import glob
import argparse

#Bokeh imports
from bokeh.plotting import figure
from bokeh.layouts import layout, widgetbox, column, row
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.models.widgets import RangeSlider, Select, TextInput,PreText, Button, DataTable, TableColumn
from bokeh.io import curdoc


#LineProfiler
#%load_ext line_profiler
#from __future__ import print_function # Always do this >:( 
#from __future__ import division

#DarkSage imports
#import matplotlib.pyplot as plt #GUI gives error when matplotlib is loaded
import itertools
from tqdm import trange
from tqdm import tqdm
import time


#DarkSage input
def galdtype_darksage(Nannuli=30):
    floattype = np.float32
    Galdesc_full = [
                    ('Type'                         , np.int32),
                    ('GalaxyIndex'                  , np.int64),
                    ('HaloIndex'                    , np.int32),
                    ('SimulationHaloIndex'          , np.int32),
                    ('TreeIndex'                    , np.int32),
                    ('SnapNum'                      , np.int32),
                    ('CentralGalaxyIndex'           , np.int64),
                    ('Mvir'                  , floattype),
                    ('mergeType'                    , np.int32),
                    ('mergeIntoID'                  , np.int32),
                    ('mergeIntoSnapNum'             , np.int32),
                    ('dT'                           , floattype),
                    ('Pos'                          , (floattype, 3)),
                    ('Vel'                          , (floattype, 3)),
                    ('Spin'                         , (floattype, 3)),
                    ('Len'                          , np.int32),
                    ('LenMax'                       , np.int32),
                    ('Mvir'                         , floattype),
                    ('Rvir'                         , floattype),
                    ('Vvir'                         , floattype),
                    ('Vmax'                         , floattype),
                    ('VelDisp'                      , floattype),
                    ('DiscRadii'                    , (floattype, Nannuli+1)), 
                    ('ColdGas'                      , floattype),
                    ('StellarMass'                  , floattype),
                    ('MergerBulgeMass'              , floattype),
                    ('InstabilityBulgeMass'          , floattype),
                    ('HotGas'                       , floattype),
                    ('EjectedMass'                  , floattype),
                    ('BlackHoleMass'                , floattype),
                    ('IntraClusterStars'            , floattype),
                    ('DiscGas'                      , (floattype, Nannuli)),
                    ('DiscStars'                    , (floattype, Nannuli)),
                    ('SpinStars'                    , (floattype, 3)),
                    ('SpinGas'                      , (floattype, 3)),
                    ('SpinClassicalBulge'           , (floattype, 3)),
                    ('StarsInSitu'                  , floattype),
                    ('StarsInstability'             , floattype),
                    ('StarsMergeBurst'              , floattype),
                    ('DiscHI'                       , (floattype, Nannuli)),
                    ('DiscH2'                       , (floattype, Nannuli)),
                    ('DiscSFR'                      , (floattype, Nannuli)), 
                    ('MetalsColdGas'                , floattype),
                    ('MetalsStellarMass'            , floattype),
                    ('ClassicalMetalsBulgeMass'     , floattype),
                    ('SecularMetalsBulgeMass'       , floattype),
                    ('MetalsHotGas'                 , floattype),
                    ('MetalsEjectedMass'            , floattype),
                    ('MetalsIntraClusterStars'      , floattype),
                    ('DiscGasMetals'                , (floattype, Nannuli)),
                    ('DiscStarsMetals'              , (floattype, Nannuli)),
                    ('SfrFromH2'                    , floattype),
                    ('SfrInstab'                    , floattype),
                    ('SfrMergeBurst'                , floattype),
                    ('SfrDiskZ'                     , floattype),
                    ('SfrBulgeZ'                    , floattype),
                    ('DiskScaleRadius'              , floattype),
                    ('CoolScaleRadius'              , floattype), 
                    ('StellarDiscScaleRadius'       , floattype),
                    ('Cooling'                      , floattype),
                    ('Heating'                      , floattype),
                    ('LastMajorMerger'              , floattype),
                    ('LastMinorMerger'              , floattype),
                    ('OutflowRate'                  , floattype),
                    ('infallMvir'                   , floattype),
                    ('infallVvir'                   , floattype),
                    ('infallVmax'                   , floattype)
                    ]
    names = [Galdesc_full[i][0] for i in range(len(Galdesc_full))]
    formats = [Galdesc_full[i][1] for i in range(len(Galdesc_full))]
    Galdesc = np.dtype({'names':names, 'formats':formats}, align=True)
    return Galdesc


#from pylab import *
import os
import routines as r
import random

# Warnings are annoying
import warnings
warnings.filterwarnings("ignore")


###### USER NEEDS TO SET THESE THINGS ######
#At the moment this is autmatically loaded when starting GUI
indir = '/fred/oz042/rdzudzar/simulation_catalogs/darksage/mini-millennium/output/' # directory where the Dark Sage data are
sim = 0 # which simulation Dark Sage has been run on -- if it's new, you will need to set its defaults below.
#   0 = Mini Millennium, 1 = Full Millennium, 2 = SMDPL

fpre = 'model_z0.000' # what is the prefix name of the z=0 files
files = range(8) # list of file numbers you want to read

Nannuli = 30 # number of annuli used for discs in Dark Sage
FirstBin = 1.0 # copy from parameter file -- sets the annuli's sizes
ExponentBin = 1.4
###### ============================== ######



##### SIMULATION DEFAULTS #####
if sim==0:
    h = 0.73
    Lbox = 62.5/h * (len(files)/8.)**(1./3)
elif sim==1:
    h = 0.73
    Lbox = 500.0/h * (len(files)/512.)**(1./3)
elif sim==2:
    h = 0.6777
    Lbox = 400.0/h * (len(files)/1000.)**(1./3)
# add here 'elif sim==3:' etc for a new simulation
else:
    print('Please specify a valid simulation.  You may need to add its defaults to this code.')
    quit()
######  ================= #####


##### READ DARK SAGE DATA #####
DiscBinEdge = np.append(0, np.array([FirstBin*ExponentBin**i for i in range(Nannuli)])) / h
G = r.darksage_snap(indir+fpre, files, Nannuli=Nannuli)
######  ================= #####


###### SET PLOTTING DEFAULTS #####
#fsize = 26
#matplotlib.rcParams.update({'font.size': fsize, 'xtick.major.size': 10, 'ytick.major.size': 10, 'xtick.major.width': 1, 'ytick.major.width': 1, 'ytick.minor.size': 5, 'xtick.minor.size': 5, 'xtick.direction': 'in', 'ytick.direction': 'in', 'axes.linewidth': 1, 'text.usetex': True, 'font.family': 'serif', 'font.serif': 'Times New Roman', 'legend.numpoints': 1, 'legend.columnspacing': 1, 'legend.fontsize': fsize-4, 'xtick.top': True, 'ytick.right': True})

NpartMed = 100 # minimum number of particles for finding relevant medians for minima on plots

outdir = './output' # where the plots will be saved
if not os.path.exists(outdir): os.makedirs(outdir)
######  =================== #####

########################################################################
#Create Pandas dataframe from the DarkSage output G['']

import pandas as pd
import numpy as np


# This is a way to converte multi dimensional data into pd.Series and then load these into the pandas dataframe
Pos = []
for p in G['Pos']:
    Pos.append(p)
Pos_df = pd.Series(Pos, dtype=np.dtype("object"))

Vel = []
for v in G['Vel']:
    Vel.append(v)
Vel_df = pd.Series(Vel, dtype=np.dtype("object"))

Spin = []
for s in G['Spin']:
    Spin.append(s)
Spin_df = pd.Series(Spin, dtype=np.dtype("object"))

Disc_r = []
for d in G['DiscRadii']:
    Disc_r.append(d)
Disc_df = pd.Series(Disc_r, dtype=np.dtype("object"))

Disc_gas = []
for g in G['DiscGas']:
    Disc_gas.append(g)
Disc_gas_df = pd.Series(Disc_gas, dtype=np.dtype("object"))

Disc_stars = []
for g in G['DiscStars']:
    Disc_stars.append(g)
Disc_stars_df = pd.Series(Disc_stars, dtype=np.dtype("object"))

SpinStars = []
for g in G['SpinStars']:
    SpinStars.append(g)
SpinStars_df = pd.Series(SpinStars, dtype=np.dtype("object"))

SpinGas = []
for g in G['SpinGas']:
    SpinGas.append(g)
SpinGas_df = pd.Series(SpinGas , dtype=np.dtype("object"))

SpinClassicalBulge = []
for g in G['SpinClassicalBulge']:
    SpinClassicalBulge.append(g)
SpinClassicalBulge_df = pd.Series(SpinClassicalBulge, dtype=np.dtype("object"))

DiscHI = []
for g in G['DiscHI']:
    DiscHI.append(g)
DiscHI_df = pd.Series(DiscHI, dtype=np.dtype("object"))

DiscH2 = []
for g in G['DiscH2']:
    DiscH2.append(g)
DiscH2_df = pd.Series(DiscH2, dtype=np.dtype("object"))

DiscSFR = []
for g in G['DiscSFR']:
    DiscSFR.append(g)
DiscSFR_df = pd.Series(DiscSFR, dtype=np.dtype("object"))

DiscGasMetals = []
for g in G['DiscGasMetals']:
    DiscGasMetals.append(g)
DiscGasMetals_df = pd.Series(DiscGasMetals, dtype=np.dtype("object"))

DiscStarsMetals = []
for g in G['DiscStarsMetals']:
    DiscStarsMetals.append(g)
DiscStarsMetals_df = pd.Series(DiscStarsMetals, dtype=np.dtype("object"))




######################################


DS = pd.DataFrame({'Type'   : G['Type'                      ],
'GalaxyIndex'               : G['GalaxyIndex'               ],
'HaloIndex'                 : G['HaloIndex'                 ],
'SimulationHaloIndex'       : G['SimulationHaloIndex'       ],
'TreeIndex'                 : G['TreeIndex'                 ],
'SnapNum'                   : G['SnapNum'                   ],
'CentralGalaxyIndex'        : G['CentralGalaxyIndex'        ],
'CentralMvir'               : G['CentralMvir'               ],
'mergeType'                 : G['mergeType'                 ],
'mergeIntoID'               : G['mergeIntoID'               ],
'mergeIntoSnapNum'          : G['mergeIntoSnapNum'          ],
'dT'                        : G['dT'                        ],
'Pos'                       : Pos_df,
'Vel'                       : Vel_df                       ,
'Spin'                      : Spin_df                      ,
'Len'                       : G['Len'                       ],
'LenMax'                    : G['LenMax'                    ],
'Mvir'                      : G['Mvir'                      ],
'Rvir'                      : G['Rvir'                      ],
'Vvir'                      : G['Vvir'                      ],
'Vmax'                      : G['Vmax'                      ],
'VelDisp'                   : G['VelDisp'                   ],
'DiscRadii'                 : Disc_df,
'ColdGas'                   : G['ColdGas'                   ],
'StellarMass'               : G['StellarMass'               ],
'MergerBulgeMass'           : G['MergerBulgeMass'           ],
'InstabilityBulgeMass'      : G['InstabilityBulgeMass'      ],
'HotGas'                    : G['HotGas'                    ],
'EjectedMass'               : G['EjectedMass'               ],
'BlackHoleMass'             : G['BlackHoleMass'             ],
'IntraClusterStars'         : G['IntraClusterStars'         ],
'DiscGas'                   : Disc_gas_df,
'DiscStars'                 : Disc_stars_df,
'SpinStars'                 : SpinStars_df,
'SpinGas'                   : SpinGas_df,
'SpinClassicalBulge'        : SpinClassicalBulge_df,
'StarsInSitu'               : G['StarsInSitu'               ],
'StarsInstability'          : G['StarsInstability'          ],
'StarsMergeBurst'           : G['StarsMergeBurst'           ],
'DiscHI'                    : DiscHI_df,
'DiscH2'                    : DiscH2_df,
'DiscSFR'                   : DiscSFR_df,
'MetalsColdGas'             : G['MetalsColdGas'             ],
'MetalsStellarMass'         : G['MetalsStellarMass'         ],
'ClassicalMetalsBulgeMass'  : G['ClassicalMetalsBulgeMass'  ],
'SecularMetalsBulgeMass'    : G['SecularMetalsBulgeMass'    ],
'MetalsHotGas'              : G['MetalsHotGas'              ],
'MetalsEjectedMass'         : G['MetalsEjectedMass'         ],
'MetalsIntraClusterStars'   : G['MetalsIntraClusterStars'   ],
'DiscGasMetals'             : DiscGasMetals_df,
'DiscStarsMetals'           : DiscStarsMetals_df,
'SfrFromH2'                 : G['SfrFromH2'                 ],
'SfrInstab'                 : G['SfrInstab'                 ],
'SfrMergeBurst'             : G['SfrMergeBurst'             ],
'SfrDiskZ'                  : G['SfrDiskZ'                  ],
'SfrBulgeZ'                 : G['SfrBulgeZ'                 ],
'DiskScaleRadius'           : G['DiskScaleRadius'           ],
'CoolScaleRadius'           : G['CoolScaleRadius'           ],
'StellarDiscScaleRadius'    : G['StellarDiscScaleRadius'    ],
'Cooling'                   : G['Cooling'                   ],
'Heating'                   : G['Heating'                   ],
'LastMajorMerger'           : G['LastMajorMerger'           ],
'LastMinorMerger'           : G['LastMinorMerger'           ],
'OutflowRate'               : G['OutflowRate'               ],
'infallMvir'                : G['infallMvir'                ],
'infallVvir'                : G['infallVvir'                ],
'infallVmax'                : G['infallVmax'                ]})

########################################################################

#Defining the axis_map
axis_map = {
        "GalaxyIndex": "GalaxyIndex",
        "CentralGalaxyIndex": "CentralGalaxyIndex",
        "StellarMass": "StellarMass",
        "ColdGas": "ColdGas",
        }


#Setting up the widgets
#First plot to appear
x_axis = Select(title="X axis", options=sorted(axis_map.keys()), value="StellarMass")
y_axis = Select(title="Y axis", options=sorted(axis_map.keys()), value="ColdGas")
codefeedback = PreText(text="",width=900)

#x=np.log10('x'*1e10/h+1),y=np.log10('y'*1e10/h+1)
#x=np.log10('x'*1e10/h+1),y=np.log10( ('y'*1e10/h+1)/('x'*1e10/h+1))

Mvir = RangeSlider(title="Mvir", start=DS["Mvir"].min(),end=5,value=(DS["Mvir"].min(),DS["Mvir"].max()),step=0.05)
ColdGas = RangeSlider(title="ColdGas", start=0.0000001,end=DS["ColdGas"].max(),value=(0.0000001,DS["ColdGas"].max()),step=0.001)
#Len = RangeSlider(title="Particles Len", start=DS["Len"].min(),end=DS["Len"].max(),value=(DS["Len"].min(),DS["Len"].max()),step=1)
SfrFromH2 = RangeSlider(title="SfrFromH2", start=DS["SfrFromH2"].min(),end=DS["SfrFromH2"].max(),value=(DS["SfrFromH2"].min(),DS["SfrFromH2"].max()),step=0.1)
StellarMass = RangeSlider(title="StellarMass", start=0.0000001,end=1,value=(0.0000001,DS["StellarMass"].max()),step=0.001)
textoutput = PreText(text="",width=300)

#Defining ColumnDataSources
mainsource=ColumnDataSource(data=dict(x=[],y=[]))
sel_datasource = ColumnDataSource(data=dict(Type=[],GalaxyIndex=[],CentralGalaxyIndex=[],
                                            StellarMass=[],ColdGas=[],Mvir=[],Len=[],HaloIndex=[],TreeIndex=[],SimulationHaloIndex=[],SnapNum=[],mergeType=[]))

#Defining Hover Tools
defaults = 'pan,box_zoom,box_select,lasso_select,reset,wheel_zoom,tap,undo,redo,save'
#hover is connected to mainsource.data below
hover_userdefined = HoverTool(tooltips=[('Type','@Type'),('HaloIndex','@HaloIndex'),('TreeIndex','@TreeIndex'),('SimulationHaloIndex','@SimulationHaloIndex'),
                                         ('SnapNum','@SnapNum'), ('mergeType','@mergeType')])
TOOLS = [defaults,hover_userdefined]

#Plots
p1 = figure(tools=TOOLS)
scatter1 = p1.circle(x='x', y='y' ,source=mainsource,size=7, color='#fa9fb5')
p2 = figure(tools=defaults)
p2.xaxis.axis_label="Stellar Mass"
p2.yaxis.axis_label="Gas mass fraction"
scatter2 = p2.circle(x='x', y='z',source=mainsource,size=7, color='#fa9fb5')



#DataTable
columns=[
		TableColumn(field="Type", title="Type"),
        TableColumn(field="GalaxyIndex", title="GalaxyIndex"),
        TableColumn(field="CentralGalaxyIndex", title="CentralGalaxyIndex"),
        TableColumn(field="StellarMass", title="StellarMass"),
        TableColumn(field="ColdGas", title="ColdGas"),
        TableColumn(field="Mvir",title="Mvir"),
        TableColumn(field="HaloIndex",title="HaloIndex"),
        TableColumn(field="TreeIndex",title="TreeIndex"),
        TableColumn(field="SimulationHaloIndex",title="SimulationHaloIndex"),
        TableColumn(field="SnapNum",title="SnapNum"),
        TableColumn(field="mergeType", title="mergeType"),
        ]
data_table = DataTable(source=sel_datasource, columns=columns, width=900, height=600)

#Filtering data
def filter_data():
    fdata = DS[
            (DS["SfrFromH2"] >= SfrFromH2.value[0]) &
            (DS["SfrFromH2"] <= SfrFromH2.value[1]) &
            (DS["StellarMass"] >= StellarMass.value[0]) &
            (DS["StellarMass"] <= StellarMass.value[1]) &
            (DS["Mvir"] >= Mvir.value[0]) &
            (DS["Mvir"] <= Mvir.value[1]) &
            #(DS["Len"] >= Len.value[0]) &
            #(DS["Len"] >= Len.value[1]) &
            (DS["ColdGas"] >= ColdGas.value[0]) &
            (DS["ColdGas"] <= ColdGas.value[1])
            ]
    return fdata

#Callback routines
def update_gui():
    codefeedback.text="Welcome to exploration of the DarkSage. You have loaded DarkSage and mini-millenium.\n The columns are : {0}".format(DS.columns.values)
    data = filter_data()
    p1.xaxis.axis_label = x_axis.value
    p1.yaxis.axis_label = y_axis.value
    x_name=axis_map[x_axis.value]
    y_name=axis_map[y_axis.value]

    #Here is the data manipulated
    mainsource.data = dict(x=np.log10(data[x_name]*1e10/h+1),y=np.log10(data[y_name]*1e10/h+1),
                            z = np.log10((data[y_name]*1e10/h+1)/(data[x_name]*1e10/h+1)),
                           Type=data["Type"],
                           GalaxyIndex=data["GalaxyIndex"],
                           CentralGalaxyIndex=data["CentralGalaxyIndex"],
                           HaloIndex=data["HaloIndex"],
                           TreeIndex=data["TreeIndex"],
                           SimulationHaloIndex=data["SimulationHaloIndex"],
                           SnapNum=data["SnapNum"],
                           mergeType=data["mergeType"],
                           ColdGas=data["ColdGas"],
                           StellarMass=data["StellarMass"],
                           Mvir=data["Mvir"])

#Selected data
def update_selected(attr,old,new):
    #inds = np.array(new['1d']['indices'])#selection for the older version than 1.1.0
    #inds = mainsource.selected.indices[0]
    print(new[0])
    inds = new[0]
    data = filter_data()

    sel_datasource.data = dict(Type=data["Type"][new[0]],SfrFromH2=data["SfrFromH2"][new[0]],CentralGalaxyIndex=data["CentralGalaxyIndex"][new[0]],
        StellarMass=data["StellarMass"][new[0]],ColdGas=data["ColdGas"][new[0]],Mvir=data["Mvir"][new[0]], HaloIndex=data["HaloIndex"][new[0]],
        TreeIndex=data["TreeIndex"][new[0]],SimulationHaloIndex=data["SimulationHaloIndex"][new[0]],SnapNum=data["SnapNum"][new[0]],
        mergeType=data["mergeType"][new[0]], GalaxyIndex=data["GalaxyIndex"][new[0]])
    textoutput.text= "Selected are \n {0} index in your dataset".format(inds)

#Callbacks for selected data points
#scatter1.data_source.on_change('selected', update_selected) 
mainsource.selected.on_change("indices", update_selected)

#Widget callbacks
controls = [x_axis,y_axis]
for control in controls:
    control.on_change('value', lambda attr,old,new: update_gui())

rangesliders = [Mvir,ColdGas,SfrFromH2,StellarMass]
for slider in rangesliders:
    slider.on_change('value', lambda attr, old, new: update_gui())

#Widgetboxes
axes = widgetbox(*controls, sizing_mode='fixed')
sliders = widgetbox(*rangesliders, sizing_mode='fixed')
data_table_input = widgetbox(data_table,sizing_mode='fixed')

#Defining the layout for the GUI
filtering = column([axes,sliders,textoutput])

#Layout
gui_layout = layout([codefeedback],
                    [filtering,p1],
                    [p2,data_table_input],sizing_mode='fixed')

update_gui()

curdoc().add_root(gui_layout)
curdoc().title = "DarkSage"

