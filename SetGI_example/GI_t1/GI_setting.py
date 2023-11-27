# 加载GI模块
# 目前直接通过sim读取含有GI模块的inp运行会报错，需要先读取无GI的模块，再加入之后运行才行
import pandas as pd
from swmm_api.input_file import read_inp_file
from swmm_api.input_file.section_labels import *
from swmm_api.input_file.sections import *
from swmm_api.input_file.helpers import *
from swmm_api.input_file.sections.others import TimeseriesData
from swmm_api.input_file.section_labels import TIMESERIES,LID_USAGE
import os
os.environ['CONDA_DLL_SEARCH_MODIFICATION_ENABLE']="1"
import numpy as np

class LID_setting(BaseSectionObject):
    """
    Cross-section geometry for street conduits.

    Section:
        [LID_USAGE]

    Purpose:
        Setting LID.

    Attributes:
        'Subcatchment'(str),'LID Process'(str),'Number'(int), 'Area'(float),'Width'(float),'InitSat'(int),
        'FromImp'(int),'ToPerv'(int),'RptFile'(str),'DrainTo'(str),'FromPerv'(int).

    Remarks:
        None
    """
    def __init__(self, subcatchment,lid_process,number,area,width,initsat,fromImp,toPerv,rptFile,drainTo,fromPerv):
        
        self.subcatchment = str(subcatchment)
        self.lid_process = str(lid_process)
        self.number = int(number)
        self.area = float(area)
        self.width = float(width)
        self.initsat = int(initsat)
        self.fromImp = int(fromImp)
        self.toPerv = int(toPerv)
        self.rptFile = str(rptFile)
        self.drainTo = str(drainTo)
        self.fromPerv = int(fromPerv)



def get_GIarea(inp,GI_rate):
    GIarea = {}
    build_area = pd.read_csv('./polygon.csv')
    for k in inp['SUBCATCHMENTS'].keys():
        #GIarea[k]=inp['SUBCATCHMENTS'][k].area*1000*building_rate*GI_rate
        GIarea[k] = build_area[build_area['Name']==k]['roofarea'].values[0]*GI_rate
    return GIarea

def set_GIusage(inp,GIarea,unit_area):
    #p = InpSection(LID_USAGE)
    p = read_inp_file(os.path.dirname(os.getcwd())+'\\SWMM\\chaohu_GI_setting.inp')[LID_USAGE]
    for k in p.keys():
        p[k].n_replicate = str(int(GIarea[k[0]]/unit_area))
        p[k].lid=k[1]
        p[k].area=unit_area
        p[k].width=0.0 
        #p[k].saturation_init,p[k].impervious_portion, p[k].route_to_pervious =0.0, 0.0, 0
        #p[k].fn_lid_report, p[k].drain_to, p[k].from_pervious='*', '*', '0'
    inp[LID_USAGE] = p
    return inp
    

