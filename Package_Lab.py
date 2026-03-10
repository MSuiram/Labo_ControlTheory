import numpy as np

import matplotlib.pyplot as plt
from IPython.display import display, clear_output

#----------------------------------------------
def LL_RT(MV,Kp,Ts,T_LEAD,T_LAG,PV,PVInit=0):
    
    """
    The function "LL_RT" needs to be included in a "for or while loop".
    
    :MV: input vector
    :Kp: process gain
    :T_LEAD: lead time constant [s]
    :T_LAG: lag time constant [s]
    :Ts: sampling period [s]
    :PV: output vector
    :PVInit: (optional: default value is 0)
    
    The function "LL_RT" appends a value to the output vector "PV".
    The appended value is obtained from a recurrent equation that depends on the discretisation method.
    """    
    
    if (T_LEAD != 0 and T_LAG != 0):
        K = Ts/T_LAG
        if len(PV) == 0:
            PV.append(PVInit)
        else: # MV[k+1] is MV[-1] and MV[k] is MV[-2]
            PV.append((1/(1+K))*PV[-1] + (K*Kp/(1+K))*((1+T_LEAD/Ts)*MV[-1] - (T_LEAD/Ts)*MV[-2]))
    else:
        PV.append(Kp*MV[-1])