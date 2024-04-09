import moduliEvoluzionePlacchette as moduli
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


cE = [27/8, -3/8, -3/4]
cB = [-9*x/8, -3*x/8, -x/8]
HEl = SparsePauliOp.from_list([("III", cE[0]), ("ZZI", cE[1]), ("IZZ", cE[1]), ("ZIZ", cE[1]), ("ZII", cE[2]), ("IZI", cE[2]), ("IIZ", cE[2])])
HMag = SparsePauliOp.from_list([("XII", cB[0]), ("XIZ", cB[1]), ("XZI", cB[1]), ("XZZ", cB[2]), 
                                ("IXI", cB[0]), ("ZXI", cB[1]), ("IXZ", cB[1]), ("ZXZ", cB[2]), 
                                ("IIX", cB[0]), ("IZX", cB[1]), ("ZIX", cB[1]), ("ZZX", cB[2])])

HQ = SparsePauliOp.sum([HEl, HMag])
H = HQ.to_matrix()

autovalori, matriceAutovettori = np.linalg.eigh(H)
matriceAutovettoriInversa = np.linalg.inv(matriceAutovettori)
nStati = len(H)

dt = T / nPunti
minP = 0

stati = [[0.7517597292134466, 0.3590777230042283, 0.35518487074876687, 0.12247950599308169, 0.36281228494087214, 0.1256877189181734, 0.1282697167325966, 0.02965972953965895],
         [-0.04629853874498452, -0.17778846086380556, 0.6782097306674496, 0.3479957902759456, -0.4613163194698939, -0.37153319354282716, 0.17657368801478737, 0.05612728266573337],
         [-0.2702498839081572, -0.39008183583157285, 0.17101908369874622, -0.009063315350084903, 0.45166864626808984, 0.2846759634059063, 0.5654109882271029, 0.3751492171882071],
         [-0.3034563956773947, 0.05240401547642042, -0.30700625651005686, 0.011101503068184842, 0.5508588534850318, 0.5544587085925963, 0.3419773216455326, 0.2880512552669651],
         [0.2171260796196633, -0.02990569712791024, -0.41368420674632983, 0.16464246870122715, 0.06494997753640629, -0.5820172369768267, 0.025965278631312535, 0.6403775370844871],
         [-0.07589054761820127, -0.2508346536818452, -0.053842488011891246, 0.6570639958707848, 0.4377259135889727, -0.05967154495362818, -0.478526120407259, -0.26932986557945776],
         [-0.02842408626525129, 0.42357618926643203, -0.12266951132288796, -0.4403755876903465, -0.260135437570755, -0.197882593461155, 0.6988286834444006, -0.12492926192459446],
         [-0.40983968525757986, 0.40035257059213664, 0.4072552520747876, -0.3465571340696492, 0.3798931952715012, -0.3154449778184111, -0.3224907611322844, 0.19484840063762243]]

#stati = np.transpose(matriceAutovettori)

for s in range(1, nStati):

    coefficienti = [0 for n in range(nStati)]
    coefficienti[0] = 0.5**0.5
    coefficienti[s] = 0.5**0.5
    vIniziale = moduli.sovrapposizione(stati, coefficienti)
    vIniziale = np.multiply(vIniziale, 1 / np.linalg.norm(vIniziale))
    print(vIniziale)

    ascisse = [(n * dt) for n in range(int(minP * nPunti), nPunti)]
    percorsoFile = "data/frequenze3PL/"
    prefissoMisure = percorsoFile + "prob3P_" + str(s)
    
    if leggiDaFile:
        probStati = moduli.leggiMisureDaFile(prefissoMisure, nStati, nPunti)

    else:
        #### Per generare i dati dal circuito
        nQubit = 3
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
            moduli.evoluzione3Pl(qc, qr, dt, x, n, nCx)
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

        plt.figure(figsize = (20, 10))
        plt.plot(ascisseEvEsatta, ordinateProb[k], "b")
        plt.plot(ascisse, probStati[k], "ro")
        plt.savefig(prefissoFigure + "Prob" + suffissoFigure)
        plt.close()

        plt.figure(figsize = (80, 10))
        plt.plot(ascisseEvEsatta, ordinateProb[k], "b")
        plt.plot(ascisse, probStati[k], "ro")
        plt.savefig(prefissoFigure + "ProbGrande" + suffissoFigure)
        plt.close()