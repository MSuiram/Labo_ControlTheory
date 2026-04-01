import pandas as pd
import json
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from matplotlib import colors as mcolors

from  package_DBR import *
from Package_Lab import *



calibration = {
  -50.0: {'alpha': 2.4162, 'gamma': 3.1981},
  -48.0: {'alpha': 2.3815, 'gamma': 3.1721},
  -46.0: {'alpha': 2.4162, 'gamma': 2.9208},
  -44.0: {'alpha': 2.5550, 'gamma': 2.8862},
  -42.0: {'alpha': 2.7401, 'gamma': 2.6783},
  -40.0: {'alpha': 3.0178, 'gamma': 2.4703},
  -38.0: {'alpha': 3.1566, 'gamma': 2.3664},
  -36.0: {'alpha': 4.2054, 'gamma': 2.0892},
  -34.0: {'alpha': 5.4240, 'gamma': 1.9159},
  -32.0: {'alpha': 20.0000, 'gamma': 1.6387}, 
  -30.0: {'alpha': 20.0000, 'gamma': 1.5347},  
  -28.0: {'alpha': 20.0000, 'gamma': 1.4231},  
  -26.0: {'alpha': 20.0000, 'gamma': 1.4231},  
  -24.0: {'alpha': 20.0000, 'gamma': 1.4231}, 
  -22.0: {'alpha': 20.0000, 'gamma': 1.3000},  
  -20.0: {'alpha': 20.0000, 'gamma': 1.1699},  
  -18.0: {'alpha': 20.0000, 'gamma': 1.0415},  
  -16.0: {'alpha': 3.9507, 'gamma': 1.0084},   
  -14.0: {'alpha': 20.0000, 'gamma': 0.7968},  
  -12.0: {'alpha': 20.0000, 'gamma': 0.6713},
  -10.0: {'alpha': 2.6195, 'gamma': 0.6613}, 
  -8.0:  {'alpha': 1.6141, 'gamma': 0.6106},   
  -6.0:  {'alpha': 1.2324, 'gamma': 0.5057},  
  -4.0:  {'alpha': 0.6938, 'gamma': 0.4451}, 
  -2.0:  {'alpha': 0.3618, 'gamma': 0.3151},  
   0.0:  {'alpha': 1.6141, 'gamma': 0.2102},   
   2.0:  {'alpha': 0.4659, 'gamma': 0.3065},  
   4.0:  {'alpha': 0.8176, 'gamma': 0.4018},   
   6.0:  {'alpha': 1.2324, 'gamma': 0.5057},  
   8.0:  {'alpha': 1.6141, 'gamma': 0.6106},   
  10.0:  {'alpha': 2.3656, 'gamma': 0.6749},   
  12.0:  {'alpha': 19.6877, 'gamma': 0.6713},
  14.0:  {'alpha': 8.2384, 'gamma': 0.8291},  
  16.0:  {'alpha': 3.9471, 'gamma': 1.0140},  
  18.0:  {'alpha': 9.4450, 'gamma': 1.0502},  
  20.0:  {'alpha': 20.0000, 'gamma': 1.1543},  
  22.0:  {'alpha': 16.5642, 'gamma': 1.2931},  
  24.0:  {'alpha': 20.0000, 'gamma': 1.3191},  
  26.0:  {'alpha': 20.0000, 'gamma': 1.3191},  
  28.0:  {'alpha': 20.0000, 'gamma': 0.1720},  
  30.0:  {'alpha': 20.0000, 'gamma': 0.1807},  
  32.0:  {'alpha': 9.9940, 'gamma': 0.0944},   
  34.0:  {'alpha': 6.4655, 'gamma': 0.0981},  
  36.0:  {'alpha': 5.1635, 'gamma': 0.0254}, 
  38.0:  {'alpha': 4.1479, 'gamma': 0.0266},  
  40.0:  {'alpha': 3.8055, 'gamma': 0.0276},  
  42.0:  {'alpha': 3.7187, 'gamma': 0.0276},  
  44.0:  {'alpha': 3.7061, 'gamma': 0.0277},  
  46.0:  {'alpha': 3.7061, 'gamma': 0.0277},  
  48.0:  {'alpha': 3.7061, 'gamma': 0.0277},  
  50.0:  {'alpha': 3.7061, 'gamma': 0.0277},   
}


def Signal_Response(alpha, gamma, sp, pv0):
    TSim = 2500
    Ts = 2
    N = int(TSim / Ts) + 1

    PV0 = pv0   # ← utilise le paramètre

    FF = True
    
    # Utilise sp et pv0 ici
    SPPath = {0: PV0, 750: sp, TSim: sp}
    DVPath = {0: 50, 1500: 60, TSim: 60}
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

    # Paramètres initiaux dépendants de pv0
    DV0 = 50
    MV0 = 50

    Kc = 0
    Ti = 0
    Td = 0

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
        if FF:
            Delay_RT(DV - DV0 * np.ones_like(DV), max(0, Thetad - Thetap), Ts, MVFFDelay)
            LL_RT(MVFFDelay, -Kd / Kp, Ts, T1p, T1d, MVFF_FLL)
            LL_RT(MVFF_FLL, 1, Ts, T2p, T2d, MVFF, 0)

        Kc, Ti, Td = IMC_TUNING(Kp, gamma, T1p, T2p)

        PID_RT(
            SP, PV, Man, MVMan, MVFF,
            Kc, Ti, Td, alpha, Ts, MVMin, MVMax,
            MV, MVP, MVI, MVD, E, ManFF, PV0   # ← PV0 cohérent
        )

        # Process P(s)
        Delay_RT(MV, Thetap, Ts, MVDelayp, MV0)
        FO_RT(MVDelayp, Kp, T1p, Ts, PV1p, 0)
        FO_RT(PV1p, 1, T2p, Ts, PV2p, 0)

        # Disturbance D(s)
        Delay_RT(DV - DV0 * np.ones_like(DV), Thetad, Ts, MVDelayd, DV0)
        FO_RT(MVDelayd, Kp, T1d, Ts, PV1d, 0)
        FO_RT(PV1d, 1, T2d, Ts, PV2d, 0)

        PV.append(PV2p[-1] + PV2d[-1] + PV0 - Kp * MV0)

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

    return results, P

#-----------------------------------

def calc_efficacité(result):
    sum = 0
    for i in range(len(result["t"])):
        sum += abs(result["SP"][i]-result["PV"][i])
    return sum

#-----------------------------------

def Optimise(alpha_range, gamma_range, nombre_points, sp,pv0):
    result = []
    
    alpha_values = np.linspace(alpha_range[0], alpha_range[1], nombre_points)
    gamma_values = np.linspace(gamma_range[0], gamma_range[1], nombre_points)
    
    for a in alpha_values:
        for g in gamma_values:
            response= Signal_Response(a, g, sp,pv0)[0]
            poid = calc_efficacité(response)
            result.append((a, g, poid))
    
    best = min(result, key=lambda x: x[2])
    print(f"alpha = {best[0]:.4f}, gamma = {best[1]:.4f}, efficacité = {best[2]:.4f}")
    return best, result


def Recursive_Optimise(alpha_range, gamma_range, nombre_points, sp, pv0, depth, final_result=None):
    if final_result is None:
        final_result = []

    best, result = Optimise(alpha_range, gamma_range, nombre_points, sp, pv0)
    final_result.append(result)
    a_best, g_best, score_best = best

    if depth == 0:
        return best, final_result

    alpha_span = (alpha_range[1] - alpha_range[0]) / 4
    gamma_span = (gamma_range[1] - gamma_range[0]) / 4

    new_alpha_range = (max(alpha_range[0], a_best - alpha_span), min(alpha_range[1], a_best + alpha_span))
    new_gamma_range = (max(gamma_range[0], g_best - gamma_span), min(gamma_range[1], g_best + gamma_span))

    new_best, final_result = Recursive_Optimise(new_alpha_range, new_gamma_range, nombre_points, sp, pv0, depth - 1, final_result)

    # ← retourne le meilleur des deux niveaux
    if new_best[2] < score_best:
        return new_best, final_result
    return best, final_result 

def plot_pv_sp(t, PV, SP, title="PV vs SP"):
    """
    Trace PV et SP en fonction du temps.
    """
    n = min(len(t) if t else 0, len(PV) if PV else 0, len(SP) if SP else 0)
    if n == 0:
        print("Données vides, rien à tracer")
        return
    
    plt.figure(figsize=(12, 6))
    plt.plot(t[:n], PV[:n], 'r-', label='PV', linewidth=2)
    plt.plot(t[:n], SP[:n], 'b--', label='SP', linewidth=2)
    plt.xlabel('Temps [s]')
    plt.ylabel('Valeur [°C]')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


#experimentation regle générale





def Optimise_Auto(alpha_range, gamma_range, nombre_points, sp, pv0, min_improvement=50, max_depth=5):
    """
    Fait le zoom récursif automatiquement, s'arrête quand l'amélioration < min_improvement pts IAE.
    """
    best, _ = Optimise(alpha_range, gamma_range, nombre_points, sp, pv0)
    a_best, g_best, prev_score = best

    for depth in range(1, max_depth):
        alpha_span = (alpha_range[1] - alpha_range[0]) / (4 ** depth)
        gamma_span = (gamma_range[1] - gamma_range[0]) / (4 ** depth)

        new_alpha_range = (max(alpha_range[0], a_best - alpha_span), min(alpha_range[1], a_best + alpha_span))
        new_gamma_range = (max(gamma_range[0], g_best - gamma_span), min(gamma_range[1], g_best + gamma_span))

        new_best, _ = Optimise(new_alpha_range, new_gamma_range, nombre_points, sp, pv0)
        a_new, g_new, new_score = new_best

        improvement = new_score - prev_score
        print(f"  depth={depth+1} → score={new_score:.2f} (amélioration={improvement:.2f} pts IAE)")

        if improvement < min_improvement:
            print(f"  ✓ Convergé à depth={depth} (amélioration < {min_improvement} pts)")
            break

        best = new_best
        a_best, g_best, prev_score = a_new, g_new, new_score
    else:
        print(f"  ⚠ max_depth={max_depth} atteint sans convergence")

    return best


def build_calibration(step_range, pv0, nombre_steps, alpha_range, gamma_range, nombre_points, min_improvement=50, max_depth=5):
    step_values = np.linspace(step_range[0], step_range[1], nombre_steps)
    calibration = {}

    for step in step_values:
        sp = pv0 + step
        print(f"Optimisation pour step={step:.1f}°C (SP={sp:.1f}, PV0={pv0})...")

        best = Optimise_Auto(alpha_range, gamma_range, nombre_points, sp, pv0, min_improvement, max_depth)

        step_key = round(float(step), 2)
        calibration[step_key] = {
            "alpha": float(best[0]),
            "gamma": float(best[1])
        }

    print("\n=== Calibration finale ===")
    print("calibration = {")
    for s, params in calibration.items():
        print(f"  {s}: {{'alpha': {params['alpha']:.4f}, 'gamma': {params['gamma']:.4f}}},")
    print("}")

    return calibration


def interpolation(sp, pv0, dico):
    step = sp - pv0

    sorted_keys = sorted(dico.keys())
    if step <= sorted_keys[0]:
        res = dico[sorted_keys[0]]
        return step, res["alpha"], res["gamma"]
    
    if step >= sorted_keys[-1]:
        res = dico[sorted_keys[-1]]
        return step, res["alpha"], res["gamma"]
    
    for i in range(len(sorted_keys) - 1):
        s0 = sorted_keys[i]
        s1 = sorted_keys[i+1]
        
        if s0 <= step <= s1:
            d0 = dico[s0]
            d1 = dico[s1]
            t = (step - s0) / (s1 - s0)
            
            alpha = d0["alpha"] + t * (d1["alpha"] - d0["alpha"])
            gamma = d0["gamma"] + t * (d1["gamma"] - d0["gamma"])
            
            return step, alpha, gamma