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

#----------------------------------------------
def PID_RT(SP, PV, Man, MVMan, MVFF, Kc, Ti, Td, alpha, Ts, MVMin, MVMax, MV, MVP, MVI, MVD, E, ManFF=False, PVInit=0, methode='EBD-EBD'):
    
    """
    The function "PID_RT" needs to be included in a "for or while loop".

    :SP: setpoint vector
    :PV: process variable vector
    :Man: manual mode vector
    :MVMan: manual value vector
    :MVFF: feedforward value vector
    :Kc: proportional gain
    :Ti: integral time
    :Td: derivative time
    :alpha: derivative filter coefficient
    :Ts: sampling period
    :MVMin: minimum manipulated variable
    :MVMax: maximum manipulated variable
    :MV: manipulated variable vector
    :MVP: proportional term vector
    :MVI: integral term vector
    :MVD: derivative term vector
    :E: error vector
    :ManFF: feedforward in manual mode flag
    :PVInit: initial process variable value
    :methode: method for discretization ('EBD-EBD' or 'TRAP-TRAP')

    The function "PID_RT" appends values to the vectors "MV", "MVP", "MVI", "MVD" and "E".
    """


    methodeI, methodeD = methode.split("-")

    if len(PV) == 0:
        E.append(SP[-1] - PVInit)
    else:
        E.append(SP[-1] - PV[-1])
    

    # Compute the proportional term
    MVP.append(Kc*E[-1])

    # Compute the integral term
    if len(MVI) == 0:
        MVI.append((Kc*Ts/Ti)*E[-1])
    else:
        if methodeI == 'EBD':
            MVI.append(MVI[-1] + (Kc*Ts/Ti)*E[-1])
        if methodeI == 'TRAP':
            MVI.append(MVI[-1] + ((Kc*Ts)/(2*Ti))*(E[-1] + E[-2]))

    # Compute the derivative term
    if len(MVD) == 0:
        MVD.append(0)
    else:
        Tfd = alpha*Td
        if methodeD == 'EBD':
            MVD.append((Tfd/(Tfd+Ts))*MVD[-1] + ((Kc*Td)/(Tfd+Ts))*(E[-1] - E[-2]))
        if methodeD == 'TRAP':
            MVD.append((Tfd-Ts/2)/(Tfd+Ts/2)*MVD[-1] + ((Kc*Td)/(Tfd+Ts/2))*(E[-1] - E[-2]))

    # Manual mode is true
    if Man[-1] == True:
        if ManFF == True:
            MVI[-1] = MVMan[-1] - MVP[-1] - MVD[-1]
        else: 
            MVI[-1] = MVMan[-1] - MVP[-1] - MVD[-1] - MVFF[-1]
    
    # Saturation
    if MVI[-1]+MVP[-1]+MVD[-1] >= MVMax:
        MVI[-1] = MVMax - MVP[-1] - MVD[-1]
    elif MVI[-1]+MVP[-1]+MVD[-1] <= MVMin:
        MVI[-1] = MVMin - MVP[-1] - MVD[-1]

    # Compute the MV
    MV.append(MVP[-1] + MVI[-1] + MVD[-1])

#----------------------------------------------
def IMC_TUNING(Kp,gamma,T1,T2=0, theta=0):
    
    """
    The function "IMC_TUNING" computes the PID parameters based on the IMC tuning rules.

    :Kp: process gain
    :gamma: constant aggressiveness
    :T1: first time constant
    :T2: second time constant (default is 0)
    :alpha: derivative filter coefficient (default is 0)

    If T2 is equal to 0, the process is considered as a first order system. 
    If Alpha is equal to 0, the process is considered as a system without delay.

    :return: Kc, Ti, Td (PID parameters)
    """
    Tc = T1* gamma

    if T2 == 0 and theta == 0: #First order
        Kc = T1/(Kp*Tc)
        Ti = T1
        Td = 0
    elif T2 == 0 and theta != 0: #First order with delay
        Kc = T1 /((Tc + theta)*Kp)
        Ti = T1
        Td = 0
    elif T2 != 0 and theta == 0: #Second order
        Kc = (T1+T2)/(Kp*Tc)
        Ti = T1 + T2
        Td = (T1*T2)/(T1 + T2)
    else: #Second order with delay
        Kc = (T1+T2)/((Tc + theta)*Kp)
        Ti = T1 + T2
        Td = (T1*T2)/(T1 + T2)       
    return Kc, Ti, Td

#----------------------------------------------
def MARGIN(Ps, omega):
    """
    The function "MARGIN" computes the gain margin and the phase margin of a system.
    :Ps: vector of the process sensitivity function evaluated at the frequencies in "omega"
    :omega: vector of frequencies at which the process sensitivity function is evaluated
    """
    # Compute the gain margin
    Index_G = np.argmin(np.abs(np.angle(Ps)+np.pi))
    Frequency_G = omega[Index_G]
    print(f"Frequency at which the phase is -180 degrees: {Frequency_G} rad/s")
    GM = 1/np.abs(Ps[Index_G])

    # Compute the phase margin
    Index_P = np.argmin(np.abs(np.abs(Ps)-1))
    Frequency_P = omega[Index_P]
    print(f"Frequency at which the gain is 1: {Frequency_P} rad/s")
    PM = 180 + np.angle(Ps[Index_P])*180/np.pi

    plt.figure(figsize = (18,12))

    plt.subplot(2,1,1)
    plt.semilogx(omega,20*np.log10(np.abs(Ps)),'cyan')
    Gain_dB = 20*np.log10(np.abs(Ps[Index_G]))
    plt.axvline(Frequency_G, color='red', linestyle='--')
    plt.axhline(0, color='black', linestyle=':')        
    plt.plot(Frequency_G, Gain_dB, 'ro')                    
    plt.xlim([np.min(omega), np.max(omega)])
    plt.ylabel('Amplitude [db]')
    plt.title('Bode plot and margins')

    plt.subplot(2,1,2)
    ph_min = np.min((180/np.pi)*np.unwrap(np.angle(Ps))) - 10
    ph_max = np.max((180/np.pi)*np.unwrap(np.angle(Ps))) + 10
    plt.semilogx(omega, (180/np.pi)*np.unwrap(np.angle(Ps)),'cyan')
    Phase_deg = (180/np.pi)*np.unwrap(np.angle(Ps))
    plt.axvline(Frequency_P, color='red', linestyle='--')   
    plt.axhline(-180, color='black', linestyle=':')         
    plt.plot(Frequency_P, Phase_deg[Index_P], 'ro')
    plt.xlim([np.min(omega), np.max(omega)])
    plt.ylim([np.max([ph_min, -200]), ph_max])
    plt.ylabel('Phase [°]')

    return GM, PM




