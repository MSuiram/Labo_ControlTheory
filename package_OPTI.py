import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from matplotlib import colors as mcolors

from  package_DBR import *
from Package_Lab import *


def Signal_Response(alpha, gamma):
    Scenario = 4
    # 1 - NO FF and Man
    # 2 - FF and Man
    # 3 - NO FF and NO Man
    # 4 - FF and NO Man

    TSim = 1000
    Ts = 1
    N = int(TSim/Ts) + 1

    # Path
    if Scenario == 1:  # 1 - NO FF and Man
        FF = False
        MVFFPath = {0: 0, TSim: 0}
        SPPath = {0: 77, TSim: 77}
        DVPath = {0: 50, 500: 60, TSim: 60}
        ManPath = {0: True, TSim: True}
        MVManPath = {0: 50, TSim: 50}

    elif Scenario == 2:  # 2 - FF and Man
        FF = True
        SPPath = {0: 77, TSim: 77}
        DVPath = {0: 50, 500: 60, TSim: 60}
        ManPath = {0: True, TSim: True}
        MVManPath = {0: 50, TSim: 50}

    elif Scenario == 3:  # 3 - NO FF and NO Man
        FF = False
        MVFFPath = {0: 0, TSim: 0}
        SPPath = {0: 77, TSim: 77}
        DVPath = {0: 50, 500: 60, TSim: 60}
        ManPath = {0: True, 225: False, TSim: False}
        MVManPath = {0: 50, 225: 0, TSim: 0}

    elif Scenario == 4:  # 4 - FF and NO Man
        FF = True
        SPPath = {0: 77, TSim: 77}
        DVPath = {0: 50, 500: 60, TSim: 60}
        ManPath = {0: True, 225: False, TSim: False}
        MVManPath = {0: 50, 225: 0, TSim: 0}

    # Listes
    t = []
    SP = []

    MV = []
    MVDelay = []

    DV = []
    MVFFDelay = []
    MVFF_FLL = []
    MVFF = []

    MVDelayp = []
    PV1p = []
    PV2p = []

    MVDelayd = []
    PV1d = []
    PV2d = []

    MVMan = []
    MVP = []
    MVI = []
    MVD = []
    PV = []
    PV_P = []
    PV_D = []
    E = []
    Man = []

    # Paramètres
    DV0 = 50
    MV0 = 50
    PV0 = 77
    gamma = gamma
    Kc = 0
    Ti = 0
    Td = 0
    alpha = alpha
    MVMin = 0
    MVMax = 100
    ManFF = False
    Kd = 0.618599605156834
    Kp = 0.540610574655048
    T1d = 147.5486385332903
    T2d = 36.22565494631446
    T1p = 130.5026136090156
    T2p = 35.46518028608689
    Thetap = 14.348568183245677
    Thetad = 13.351904889062693
    

    for i in range(0, N):

        t.append(i * Ts)
        SelectPath_RT(SPPath, t, SP)
        SelectPath_RT(DVPath, t, DV)
        SelectPath_RT(ManPath, t, Man)
        SelectPath_RT(MVManPath, t, MVMan)

        # MVFF
        if FF is True:
            Delay_RT(DV - DV0 * np.ones_like(DV), max(0, Thetad - Thetap), Ts, MVFFDelay)
            LL_RT(MVFFDelay, -Kd / Kp, Ts, T1p, T1d, MVFF_FLL)
            LL_RT(MVFF_FLL, 1, Ts, T2p, T2d, MVFF, 0)
        else:
            SelectPath_RT(MVFFPath, t, MVFF)

        Kc, Ti, Td = IMC_TUNING(Kp, gamma, T1p, T2p)

        PID_RT(
            SP, PV, Man, MVMan, MVFF,
            Kc, Ti, Td, alpha, Ts, MVMin, MVMax,
            MV, MVP, MVI, MVD, E, ManFF, PV0
        )

        # P(s)
        Delay_RT(MV, Thetap, Ts, MVDelayp, MV0)
        FO_RT(MVDelayp, Kp, T1p, Ts, PV1p, 0)
        FO_RT(PV1p, 1, T2p, Ts, PV2p, 0)

        # D(s)
        Delay_RT(DV - DV0 * np.ones_like(DV), Thetad, Ts, MVDelayd, DV0)
        FO_RT(MVDelayd, Kp, T1d, Ts, PV1d, 0)
        FO_RT(PV1d, 1, T2d, Ts, PV2d, 0)

        PV.append(PV2p[-1] + PV2d[-1] + PV0 - Kp * MV0)

    # Construire le dictionnaire de toutes les listes
    results = {
        "t": t,
        "SP": SP,
        "MV": MV,
        "MVDelay": MVDelay,
        "DV": DV,
        "MVFFDelay": MVFFDelay,
        "MVFF_FLL": MVFF_FLL,
        "MVFF": MVFF,
        "MVDelayp": MVDelayp,
        "PV1p": PV1p,
        "PV2p": PV2p,
        "MVDelayd": MVDelayd,
        "PV1d": PV1d,
        "PV2d": PV2d,
        "MVMan": MVMan,
        "MVP": MVP,
        "MVI": MVI,
        "MVD": MVD,
        "PV": PV,
        "PV_P": PV_P,
        "PV_D": PV_D,
        "E": E,
        "Man": Man,
    }
    P = Process({'Kp': Kp, 'Tlag1': T1p, 'Tlag2': T2p, 'theta': Thetap})
    return results,P

#-----------------------------------

def find_settling_index(PV, SP, window=10, tol_percent=1.0, check_until_end=True):
    """
    Trouve le premier indice où PV reste dans la bande de tolérance
    pendant `window` points consécutifs.

    Si check_until_end=True (défaut) : vérifie aussi que PV ne quitte
    PLUS JAMAIS la bande après ce point → élimine les faux settlings
    des systèmes instables à phase stable transitoire.

    Retourne l'indice, ou None si pas de settling stable.
    """
    n = len(PV)
    if n == 0 or len(SP) != n or n < window:
        return None

    for i in range(0, n - window + 1):

        # 1. Vérifier la fenêtre de `window` points consécutifs
        window_ok = True
        for k in range(window):
            pv = PV[i + k]
            sp = SP[i + k]
            if sp == 0:
                window_ok = False
                break
            tol = abs(sp) * tol_percent / 100.0
            if not (sp - tol <= pv <= sp + tol):
                window_ok = False
                break

        if not window_ok:
            continue

        # 2. Vérifier que PV reste dans la bande jusqu'à la fin
        if check_until_end:
            tail_ok = True
            for j in range(i + window, n):
                pv = PV[j]
                sp = SP[j]
                if sp == 0:
                    tail_ok = False
                    break
                tol = abs(sp) * tol_percent / 100.0
                if not (sp - tol <= pv <= sp + tol):
                    tail_ok = False
                    break

            if not tail_ok:
                continue  # faux settling, on continue la recherche

        return i  # settling confirmé

    return None  # instable ou jamais convergé

#-----------------------------------

def plot_pv_sp_with_index(t, PV, SP, idx, title="PV vs SP"):
    """
    Si idx=None ou invalide → trace juste PV/SP sans trait vertical
    """
    # Sécurité robuste
    n = min(len(t) if t else 0, len(PV) if PV else 0, len(SP) if SP else 0)
    
    if n == 0:
        print("Données vides, rien à tracer")
        return
    
    plt.figure(figsize=(12, 6))
    
    # Tracer PV et SP toujours
    plt.plot(t[:n], PV[:n], 'r-', label='PV', linewidth=2)
    plt.plot(t[:n], SP[:n], 'b--', label='SP', linewidth=2)
    
    # Trait vertical SEULEMENT si idx valide
    if idx is not None and 0 <= idx < n:
        plt.axvline(x=t[idx], color='k', linestyle=':', linewidth=3, 
                   label=f'Settling point (t={t[idx]:.0f}s)')
        print(f"Trait vertical ajouté à t={t[idx]:.0f}s (index {idx})")
    else:
        print(f"Index {idx} ignoré (invalide), pas de trait vertical")
    
    plt.xlabel('Temps [s]')
    plt.ylabel('Valeur [°C]')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

#----------------------------------------------

def Optimise(step=0.1):
    values = []
    alphas = np.arange(0.1, 5.0, step)
    gammas = np.arange(0.1, 5.0, step)
    for i in alphas:
        for j in gammas:
            results, P = Signal_Response(i, j)
            idx = find_settling_index(results["PV"], results["SP"], window=10, tol_percent=0.1, check_until_end=True)
            if idx is not None:
                print(f"Settling stable trouvé pour alpha={i}, gamma={j} à t={results['t'][idx]:.0f}s")
                print(idx)
                values.append((i, j, results['t'][idx]))
            else:
                print(f"Pas de settling stable pour alpha={i}, gamma={j}")
    
    values.sort()  # Trie par settling_time croissant
    best_time, best_alpha, best_gamma = values[0]
    
    print(f"*** MEILLEUR: alpha={best_alpha:.3f}, gamma={best_gamma:.3f}, time={best_time:.0f}s ***")
    return best_alpha, best_gamma