# 加载GI模块
# 目前直接通过sim读取含有GI模块的inp运行会报错，需要先读取无GI的模块，再加入之后运行才行
import pandas as pd
from swmm_api.input_file import read_inp_file
from swmm_api.input_file.section_labels import *
from swmm_api.input_file.sections import *
from swmm_api.input_file.helpers import *
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
        