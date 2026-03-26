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

    # Proportional term
    MVP.append(Kc * E[-1])

    # Integral term
    if len(MVI) == 0:
        MVI.append((Kc * Ts / Ti) * E[-1])
    else:
        if methodeI == 'EBD':
            MVI.append(MVI[-1] + (Kc * Ts / Ti) * E[-1])
        if methodeI == 'TRAP':
            MVI.append(MVI[-1] + ((Kc * Ts) / (2 * Ti)) * (E[-1] + E[-2]))

    # Derivative term
    if len(MVD) == 0:
        MVD.append(0)
    else:
        Tfd = alpha * Td
        if methodeD == 'EBD':
            MVD.append((Tfd / (Tfd + Ts)) * MVD[-1] + ((Kc * Td) / (Tfd + Ts)) * (E[-1] - E[-2]))
        if methodeD == 'TRAP':
            MVD.append((Tfd - Ts/2) / (Tfd + Ts/2) * MVD[-1] + ((Kc * Td) / (Tfd + Ts/2)) * (E[-1] - E[-2]))

    # Manual mode
    if Man[-1] == True:
        if ManFF == True:
            MVI[-1] = MVMan[-1] - MVP[-1] - MVD[-1] - MVFF[-1]
        else:
            MVI[-1] = MVMan[-1] - MVP[-1] - MVD[-1]

    # Saturation — PID terms clamped to [MVMin-MVFF, MVMax-MVFF]
    if MVI[-1] + MVP[-1] + MVD[-1] + MVFF[-1]>= MVMax :
        MVI[-1] = MVMax - MVFF[-1] - MVP[-1] - MVD[-1]
    elif MVI[-1] + MVP[-1] + MVD[-1] + MVFF[-1]<= MVMin :
        MVI[-1] = MVMin - MVFF[-1] - MVP[-1] - MVD[-1]

    # MV output includes MVFF
    MV.append(MVP[-1] + MVI[-1] + MVD[-1] + MVFF[-1])

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
def MARGIN(Ps, Cs, omega):
    """
    Computes and plots the gain margin, phase margin, and -3dB bandwidth of a system.
    :Ps: vector of the process transfer function evaluated at frequencies in omega
    :Cs: vector of the controller transfer function evaluated at frequencies in omega
    :omega: vector of frequencies [rad/s]
    """
    Ls = Cs * Ps
    gain_dB = 20 * np.log10(np.abs(Ls))
    phase_unwrapped = np.unwrap(np.angle(Ls)) * 180 / np.pi

    # Gain margin: phase = -180°
    Index_G = np.argmin(np.abs(phase_unwrapped + 180))
    Frequency_G = omega[Index_G]
    GM = 20 * np.log10(1 / np.abs(Ls[Index_G]))  # dB

    # Phase margin: gain = 0 dB
    Index_P = np.argmin(np.abs(np.abs(Ls) - 1))
    Frequency_P = omega[Index_P]
    PM = 180 + phase_unwrapped[Index_P]

    # Bandwidth: gain = -3 dB
    Index_BW = np.argmin(np.abs(gain_dB + 3))
    Frequency_BW = omega[Index_BW]

    print(f"Phase crossover frequency : {Frequency_G:.4f} rad/s")
    print(f"Bandwidth (-3dB)          : {Frequency_BW:.4f} rad/s")
    print(f"Gain Margin               : {GM:.2f} dB")
    print(f"Phase Margin              : {PM:.2f} °")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12))

    # --- Gain plot ---
    ax1.semilogx(omega, gain_dB, 'cyan')
    ax1.axhline(-3, color='black', linestyle=':')
    ax1.axvline(Frequency_BW, color='red', linestyle='--', label=f'ωBW = {Frequency_BW:.3f} rad/s')
    ax1.plot(Frequency_BW, -3, 'ro')
    ax1.set_xlim([np.min(omega), np.max(omega)])
    ax1.set_ylabel('Amplitude [dB]')
    ax1.set_title('Bode plot and margins')
    ax1.legend()

    # --- Phase plot ---
    ph_min = np.min(phase_unwrapped) - 10
    ph_max = np.max(phase_unwrapped) + 10
    ax2.semilogx(omega, phase_unwrapped, 'cyan')
    ax2.axhline(-180, color='black', linestyle=':')
    ax2.axvline(Frequency_G, color='red', linestyle='--', label=f'ω180 = {Frequency_G:.3f} rad/s')
    ax2.plot(Frequency_G, phase_unwrapped[Index_G], 'ro')
    ax2.set_xlim([np.min(omega), np.max(omega)])
    ax2.set_ylim([np.max([ph_min, -400]), ph_max])
    ax2.set_ylabel('Phase [°]')
    ax2.set_xlabel('Fréquence [rad/s]')
    ax2.legend()

    plt.tight_layout()
    plt.show()

    return GM, PM, Frequency_BW