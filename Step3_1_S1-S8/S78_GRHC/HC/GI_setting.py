# 加载GI模块
# 目前直接通过sim读取含有GI模块的inp运行会报错，需要先读取无GI的模块，再加入之后运行才行
import pandas as pd
from swmm_api.input_file import read_inp_file
from swmm_api.input_file.section_labels import *
from swmm_api.input_file.sections import *
from swmm_api.input_file.helpers import *
import numpy as np

### 获取建筑屋顶面积，设置GI，保存inp
def add_GI(inp):
    roof=pd.read_csv('./SWMM/polygon.csv')
    unit_area=10
    for i in range(roof.shape[0]):
        number=int(0.5*roof[['roofarea']].iloc[i][0]/unit_area)
        #print(roof[['Name']].iloc[i][0],number)
        # added line of LID_USAGE
        item=LID_setting(roof[['Name']].iloc[i][0],'GreenRoof',number,10,0,0,0,0,'*','*',0)
        inp[LID_USAGE][roof[['Name']].iloc[i][0]]=item

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
        