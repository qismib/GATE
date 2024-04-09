import moduliEvoluzionePlacchette as moduli
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.providers.basicaer import QasmSimulatorPy
from qiskit.quantum_info import SparsePauliOp
from qiskit import transpile

x = 1.5
T = 40
nPunti = 400
leggiDaFile = True
trotterizzazione = "Suzuki" # "BM", "Suzuki", "Verlet"

H = [[0, -2*x, -2*x, 0],
     [-2*x, 3, 0, -x/2],
     [-2*x, 0, 3, -x/2],
     [0, -x/2, -x/2, 3]]

autovalori, matriceAutovettori = np.linalg.eigh(H)
matriceAutovettoriInversa = np.linalg.inv(matriceAutovettori)
nStati = len(H)

dt = T / nPunti
minP = 0

stati = [[0.808992047467855, 0.4101442096668511, 0.4101115648454173, 0.09550967904126201],
         [-0.20200917702552543, 0.30548582993873763, -0.12773951931771707, 0.9217121651082778],
         [-0.06861852324779168, -0.6452995532134647, 0.7024301247942335, 0.29235578439865517],
         [0.5480385666960621, -0.5706671071253255, -0.5663434088132931, 0.23075512031735548]]

for s in range(1, nStati):

    coefficienti = [0 for n in range(nStati)]
    coefficienti[0] = 0.5**0.5
    coefficienti[s] = 0.5**0.5
    vIniziale = moduli.sovrapposizione(stati, coefficienti)
    vIniziale = np.multiply(vIniziale, 1 / np.linalg.norm(vIniziale))
    print(vIniziale)

    ascisse = [(n * dt) for n in range(int(minP * nPunti), nPunti)]
    percorsoFile = "data/frequenze2PL/"
    prefissoMisure = percorsoFile + "prob2P_" + str(s)

    if leggiDaFile:
        probStati = moduli.leggiMisureDaFile(prefissoMisure, nStati, nPunti)

    else:
        #### Per generare i dati dal circuito
        nQubit = 2
        nStati = 2**nQubit
        statiAttesi = [f"{k:0{nQubit}b}" for k in range(nStati)]
        qr = QuantumRegister(nQubit)
        cr = ClassicalRegister(nQubit)
        qc = QuantumCircuit(qr, cr)
        backend = QasmSimulatorPy()
        ripetizioni = 10000
        nCx = 1

        probStati = [[] for k in range(nStati)]
        for n in range(int(minP * nPunti), nPunti):
            qc.prepare_state(vIniziale)
            moduli.evoluzione2Pl(qc, qr, dt, x, n, nCx, tipo = trotterizzazione)
            qc.measure(qr, cr)

            circuitoTradotto = transpile(qc, backend)
            res = backend.run(circuitoTradotto, shots = ripetizioni).result().get_counts(qc)
            misure = moduli.dizionarioInLista(res, statiAttesi)

            for k in range(nStati):
                probStati[k].append(misure[k] / ripetizioni)

            qc.clear()

        moduli.scriviMisureSuFile(probStati, prefissoMisure)
        ###

    #### Evoluzione esatta
    nPuntiEsatta = 4000
    ascisseEvEsatta = [(T * n / nPuntiEsatta) for n in range(nPuntiEsatta)]
    ordinateProb = [[] for n in range(len(H))]
    for t in ascisseEvEsatta:
        stato = moduli.evoluzioneHamiltoniana(vIniziale, autovalori, matriceAutovettori, matriceAutovettoriInversa, t)
        for m in range(len(vIniziale)):
            ordinateProb[m].append(abs(stato[m]) ** 2)

    #moduli.scriviMisureSuFile(ordinateProb, prefissoMisure)

    #### Grafici
    for k in range(nStati):
        prefissoFigure = percorsoFile
        suffissoFigure = "_" + str(s) + "_" + str(k) + ".png"

        font = {'family' : 'serif',
                'weight' : 'normal',
                'size'   : 40}

        matplotlib.rc('font', **font)

        plt.figure(figsize = (15, 10))
        plt.subplots_adjust(top = 0.98, bottom = 0.18, left = 0.15, right = 0.98)
        plt.xlabel("t [$2 / g^2$]", labelpad = 20)
        plt.ylabel("Probability", labelpad = 20)
        plt.plot(ascisseEvEsatta, ordinateProb[k], "b", linewidth = 3)
        plt.plot(ascisse, probStati[k], "ro", markersize = 20)
        plt.xlim(35, 40)
        plt.savefig(prefissoFigure + "Prob" + suffissoFigure)
        plt.close()

        plt.figure(figsize = (80, 10))
        plt.plot(ascisseEvEsatta, ordinateProb[k], "b")
        plt.plot(ascisse, probStati[k], "ro")
        plt.savefig(prefissoFigure + "ProbGrande" + suffissoFigure)
        plt.close()