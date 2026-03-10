import numpy as np

import matplotlib.pyplot as plt
from IPython.display import display, clear_output

#-----------------------------------
def myRound(x, base=5):
    
    """
    Returns a float that is the closest multiple of "base" near "x"
    Based on: https://stackoverflow.com/questions/2272149/round-to-5-or-other-number-in-python
    
    :x: parameter that is rounded to a multiple of "base"
    :base: "base" parameter (optional: default value is 5)
    
    :return: rounded parameter
    """
    
    return float(base * round(float(x)/base))

#-----------------------------------
def SelectPath_RT(path,time,signal):
    
    """
    The function "SelectPath_RT" needs to be included in a "for or while loop".
    
    :path: dictionary input describing a path in time. Example: path = {0: 0, 5: 1, 50: 2, 80: 3, 100: 3}
    :time: time vector.
    :signal: signal vector that is being constructed using the input "path" and the vector "time".
    
    The function "SelectPath_RT" takes the last element in the vector "time" and, given the input "path", it appends the correct value to the vector "signal".
    """    
    
    for timeKey in path:
        if(time[-1] >= timeKey):
            timeKeyPrevious = timeKey    
    
    value = path[timeKeyPrevious]
    signal.append(value)

#-----------------------------------
def Delay_RT(MV,theta,Ts,MV_Delay,MVInit=0):
    
    """
    The function "Delay_RT" needs to be included in a "for or while loop".
    
    :MV: input vector
    :theta: delay [s]
    :Ts: sampling period [s]
    :MV_Delay: delayed input vector
    :MVInit: (optional: default value is 0)
    
    The function "Delay_RT" appends a value to the vector "MV_Delay".
    The appended value corresponds to the value in the vector "MV" "theta" seconds ago.
    If "theta" is not a multiple of "Ts", "theta" is replaced by Ts*int(np.ceil(theta/Ts)), i.e. the closest multiple of "Ts" larger than "theta".
    If the value of the vector "input" "theta" seconds ago is not defined, the value "MVInit" is used.
    """
    
    NDelay = int(np.ceil(theta/Ts))
    if NDelay > len(MV)-1:
        MV_Delay.append(MVInit)
    else:    
        MV_Delay.append(MV[-NDelay-1])

#-----------------------------------        
def FO_RT(MV,Kp,T,Ts,PV,PVInit=0,method='EBD'):
    
    """
    The function "FO_RT" needs to be included in a "for or while loop".
    
    :MV: input vector
    :Kp: process gain
    :T: lag time constant [s]
    :Ts: sampling period [s]
    :PV: output vector
    :PVInit: (optional: default value is 0)
    :method: discretisation method (optional: default value is 'EBD')
        EBD: Euler Backward difference
        EFD: Euler Forward difference
        TRAP: Trapezoïdal method
    
    The function "FO_RT" appends a value to the output vector "PV".
    The appended value is obtained from a recurrent equation that depends on the discretisation method.
    """    
    
    if (T != 0):
        K = Ts/T
        if len(PV) == 0:
            PV.append(PVInit)
        else: # MV[k+1] is MV[-1] and MV[k] is MV[-2]
            if method == 'EBD':
                PV.append((1/(1+K))*PV[-1] + (K*Kp/(1+K))*MV[-1])
            elif method == 'EFD':
                PV.append((1-K)*PV[-1] + K*Kp*MV[-2])
            elif method == 'TRAP':
                PV.append((1/(2*T+Ts))*((2*T-Ts)*PV[-1] + Kp*Ts*(MV[-1] + MV[-2])))            
            else:
                PV.append((1/(1+K))*PV[-1] + (K*Kp/(1+K))*MV[-1])
    else:
        PV.append(Kp*MV[-1])

#-----------------------------------
def LEAD_LAG_RT(MV,Kp,Ts,T_LEAD,T_LAG,PV,PVInit=0,method='EBD'):
    
    """
    The function "LEAD_LAG_RT" needs to be included in a "for or while loop".
    
    :MV: input vector
    :Kp: process gain
    :T_LEAD: lead time constant [s]
    :T_LAG: lag time constant [s]
    :Ts: sampling period [s]
    :PV: output vector
    :PVInit: (optional: default value is 0)
    :method: discretisation method (optional: default value is 'EBD')
        EBD: Euler Backward difference
        EFD: Euler Forward difference
        TRAP: Trapezoïdal method
    
    The function "LEAD_LAG_RT" appends a value to the output vector "PV".
    The appended value is obtained from a recurrent equation that depends on the discretisation method.
    """    
    
    if (T_LEAD != 0 and T_LAG != 0):
        K = Ts/T_LAG
        if len(PV) == 0:
            PV.append(PVInit)
        else: # MV[k+1] is MV[-1] and MV[k] is MV[-2]
            PV.append((1/(1+K))*PV[-1] + (K*Kp/(1+K))((1+T_LEAD/T_LAG)*MV[-1]-(T_LEAD/T_LAG)*MV[-2]))
    else:
        PV.append(Kp*MV[-1])