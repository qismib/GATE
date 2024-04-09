import numpy as np
from scipy.optimize import minimize
from noisyopt import minimizeCompass, minimizeSPSA
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import transpile
from qiskit.providers.fake_provider import FakeLagos
from qiskit.primitives import Estimator
from qiskit_aer import QasmSimulator
from qiskit_aer.primitives import Estimator as AerEstimator
from qiskit_aer.noise import NoiseModel
from qiskit.quantum_info import SparsePauliOp
import moduliHR as simCl
from valoreAspettazionePauli import valoreAspettazioneOp as valoreAspPauli
from valoreAspettazionePauli import mediaVarianzaOp as misuraValoreMedioVarianza
import time

#### Con circuiti quantistici
def preparaStatoReale(parametri, circuito, qr):
    nQubit = qr.size
    gdl = 2**nQubit
    if len(parametri) != gdl:
        raise Exception("preparaStatoReale(): i parametri forniti", len(parametri), "non corrispondono al numero di gradi di libertà necessari", gdl)
    
    circuito.prepare_state(parametri, qr)

def misuraValoreMedioVarianzaStimatore(circuito, operatore, stimatore, ripetizioni = 1000):
    res = stimatore.run(circuits = [circuito], observables = [operatore], parameter_values = [], shots = ripetizioni)
    return [res.result().values[0], res.result().metadata[0]["variance"]]

# -Performance-
# Senza transpile (1000 circuiti)
# - Unica chiamata a estimator.run() con tutti i circuiti: ~ 0.06 s
# - Una chiamata a estimator.run() per circuito (1000): ~ 9 s
# 
# Con transpile (1000 circuiti)
# - Unica chiamata a estimator.run() con tutti i circuiti: ~ 33 s
# - Una chiamata a estimator.run() per circuito (1000): ~ 40 s
# 
# L'andamento con il numero di circuiti è sostanzialmente lineare

# N = 2
# qr = QuantumRegister(N)
# cr = ClassicalRegister(N)
# qc = QuantumCircuit(qr, cr)
# backend = QasmSimulator()

# parametri = [ 0.45058741, -0.0451177,  -0.43045355, -0.78079775]
# parametri = np.multiply(parametri, 1 / np.linalg.norm(parametri))

# x = 1.5
# c = [21/8, -9/8, -3/8, -9/8, -3*x/2, -x/2, -3*x/2, -x/2]
# H = SparsePauliOp.from_list([("II", c[0]), ("ZI", c[1]), ("ZZ", c[2]), ("IZ", c[3]), ("XI", c[4]), ("XZ", c[5]), ("IX", c[6]), ("ZX", c[7])])
# estimator = Estimator()

# circuiti = []
# oss = []
# par = []

# M = 1000
# ti = time.time()
# for n in range(M):
#     preparaStatoReale(parametri, qc, qr)
#     circuito = transpile(qc, backend)
#     #circuito.draw(output = "mpl", filename = "spettroGauge/circ.png")
#     circuiti.append(circuito)
#     oss.append(H)
#     par.append([])
#     qc.clear()
#     #res = misuraValoreMedioVarianzaStimatore(circuito, H, estimator)
    
# res = estimator.run(circuits = circuiti, observables = oss, parameter_values = par, shots = 10000)
    
# print(M, "esecuzioni in", (time.time() - ti), "secondi")

# ti = time.time()
# for n in range(M):
#     preparaStatoReale(parametri, qc, qr)
#     circuito = transpile(qc, backend)
#     #circuito.draw(output = "mpl", filename = "spettroGauge/circ.png")
#     qc.clear()
#     res = misuraValoreMedioVarianzaStimatore(circuito, H, estimator)

# print(res, M, "esecuzioni in", (time.time() - ti), "secondi")

_nCircuiti_ = 0
_nDuplicati_ = 0
_statoAttuale_ = 0

def stopMinimizzazione(stato, varianza, costo, soglia, soppressione):
    global _statoAttuale_
    sogliaCosto = soppressione * soglia
    if (abs(varianza) < soglia) and (costo < sogliaCosto):
        _statoAttuale_ = stato
        raise ValueError
    
    if (abs(varianza) < (3 * soglia)) and costo > soppressione:
        _statoAttuale_ = stato
        raise InterruptedError


def funzioneCostoSuCircuito(parametri, hamiltoniana, nStati, statiInferiori, soppressione = 20, stimatore = None):
    nQubit = int(np.log2(nStati))
    if nQubit != int(nQubit):
        raise Exception("Lo stato non può essere rappresentato con qubit")
    
    parametri = np.multiply(parametri, (1 / np.linalg.norm(parametri)))

    stato = simCl.statoGenericoReale(parametri)
    sogliaVar = 0.05
    energia = simCl.valoreMedio(stato, hamiltoniana.to_matrix())
    varianzaCl = simCl.varianza(stato, hamiltoniana.to_matrix()) # sono abbastanza simili a quella su circuiti
    
    global _nCircuiti_
    _nCircuiti_ += 1

    qr = QuantumRegister(nQubit)
    cr = ClassicalRegister(nQubit)
    qc = QuantumCircuit(qr, cr)

    preparaStatoReale(parametri, qc, qr)
    energia = np.real(valoreAspPauli(hamiltoniana, qc))
    #energia, varianza = misuraValoreMedioVarianza(hamiltoniana, qc) # misuraValoreMedioVarianza(qc, hamiltoniana, stimatore, ripetizioni)
    qc.clear()

    costo = energia #varianza
    for i in range(len(statiInferiori)):
        sovrapposizione = np.vdot(statiInferiori[i], stato)
        costo += soppressione * abs(sovrapposizione)
    
    #stopMinimizzazione(stato, varianzaCl, costo, sogliaVar, soppressione)

    #print(_nCircuiti_, energia, varianza, stato, costo, soppressione)
    print(_nCircuiti_, energia, varianzaCl, stato, costo, soppressione)
    return costo


def wrapperMinimizzazione(parametri, hamiltoniana, nStati, statiInferiori, soppressione = 20, stimatore = None):
    global _statoAttuale_
    try:
        stato = minimizeSPSA(func = funzioneCostoSuCircuito, 
                             x0 = parametri, 
                             args = (hamiltoniana, nStati, statiInferiori, soppressione, stimatore),
                             paired = False,
                             a = 2,
                             niter = 800) #1000
        return stato.x 
    
    except ValueError:
        print("Trovato uno stato buono")
        stato = _statoAttuale_
        _statoAttuale_ = 0

        return stato
    
    except InterruptedError:
        print("Stato troppo vicino ad uno già trovato")
        stato = _statoAttuale_
        _statoAttuale_ = 0

        return stato
    

# x = 1.5
# c = [21/8, -9/8, -3/8, -9/8, -3*x/2, -x/2, -3*x/2, -x/2]
# H = SparsePauliOp.from_list([("II", c[0]), ("ZI", c[1]), ("ZZ", c[2]), ("IZ", c[3]), ("XI", c[4]), ("XZ", c[5]), ("IX", c[6]), ("ZX", c[7])])

# par = [0, 0.5**0.5, -0.5**0.5, 0] # autostato per ogni x
# # par = [0.45058741, -0.0451177,  -0.43045355, -0.78079775] # quasi autostato per x = 1.5
# # par = np.multiply(par, 1 / np.linalg.norm(par))
# ti = time.time()

# M = 1000
# for n in range(M):
#     x = funzioneCostoSuCircuito(par, H, 4, [])
# print(x, M, _nCircuiti_, "esecuzioni in", (time.time() - ti), "secondi")

def trovaAutostato(hamiltoniana, nStati, statiInferiori, stimatore):
    rng = np.random.default_rng()
    copiaStatiInferiori = statiInferiori.copy()

    trovatoNuovoMinimo = False
    soppressione = 20

    while not trovatoNuovoMinimo:
        parametriIniziali = rng.random((nStati,))
        parametriIniziali = np.add(parametriIniziali, -0.5)
        parametriIniziali = np.multiply(parametriIniziali, (1 / np.linalg.norm(parametriIniziali)))
        parametriIniziali = simCl.ortogonalizza(parametriIniziali, statiInferiori)
        print("\nIn ingresso di minimizzazione:", parametriIniziali)
        
        parametriMinimi = wrapperMinimizzazione(parametriIniziali, hamiltoniana, nStati, copiaStatiInferiori, soppressione, stimatore)

        varianzaMassima = 0.4
        entrateMinime = parametriMinimi
        stato = simCl.statoGenericoReale(entrateMinime)
        var = simCl.varianza(stato, hamiltoniana)
        if var > varianzaMassima:
            print("trovaAutostato(): varianza troppo alta =", var, "Stato =", stato)
            continue
        
        trovatoNuovoMinimo = True
        print("trovaAutostato(): minimizzazione terminata. Varianza =", var, "Stato =", stato)
        for i in range(len(statiInferiori)):
            sogliaSovr = 0.1
            sovrapposizione = abs(np.vdot(statiInferiori[i], stato))
            if sovrapposizione > sogliaSovr:
                print("trovaAutostato(): individuato uno stato troppo vicino agli altri autostati. Si ripete la minimizzazione\n", i, sovrapposizione)
                #soppressione += 10
                trovatoNuovoMinimo = False
                global _nDuplicati_
                _nDuplicati_ += 1
                break

    return stato

def trovaTuttiAutostati(hamiltoniana, nStati, rumore):
    autostati = []
    stimatore = None
    if rumore:
        backend = FakeLagos()
        modelloRumore = NoiseModel.from_backend(backend)
        mappaQubit = backend.configuration().coupling_map
        ripetizioni = 1000
        stimatore = AerEstimator(
            approximation = True,
            backend_options = {"coupling_map" : mappaQubit, "noise_model" : modelloRumore}, 
            #transpile_options = {"optimization_level" : 0}, 
            run_options = {"shots" : ripetizioni})

    for i in range(nStati):
        stato = trovaAutostato(hamiltoniana, nStati, autostati, stimatore)
        autostati.append(stato)
        simCl.salvaBackupStato("spettroGauge/backupMP.txt", stato)

    return autostati


def trovaTuttiAutostatiMP(hamiltoniana, nStati, rumore, autovettoriMP):
    autovettori = trovaTuttiAutostati(hamiltoniana, nStati, rumore)

    for i in range(len(autovettori)):
        autovettoriMP[i] = autovettori[i]