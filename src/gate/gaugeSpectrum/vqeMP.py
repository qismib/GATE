import numpy as np
import moduliHR as moduli
import moduliHRcircuiti as simQ
from qiskit.quantum_info import SparsePauliOp
import multiprocessing as mp
import time

d = 4 #2, 4, 8
x = 1.5
nMisureSpettro = 1
simulazioneQuantistica = True

def misuraSpettroClassica(nMisureSpettro, H):
    autovettori = []

    if __name__ == "__main__":
        mp.set_start_method("fork")
        processi = []
        nProc = nMisureSpettro
        manager = mp.Manager()
        tuttiAutovettori = []
        dim = len(H)
        for i in range(nProc):
            tuttiAutovettori.append(manager.list())
            for n in range(dim):
                tuttiAutovettori[i].append(manager.list())
            
            pr = mp.Process(target = moduli.trovaTuttiAutostatiMP, args = (H, tuttiAutovettori[i]))
            processi.append(pr)
            pr.start()
        
        for i in range(nProc):
            processi[i].join()
        
        for i in range(nProc):
            autovettori.append([])
            for j in range(dim):
                autovettori[i].append([])
                for k in range(dim):
                    autovettori[i][j].append(tuttiAutovettori[i][j][k])
                    #print(i, j, k, tuttiAutovettori[i][j][k])
    
    #print("autovettori", autovettori)
    return autovettori


def misuraSpettroCircuito(nMisureSpettro, HQ, nStati, rumore):
    autovettori = []

    if __name__ == "__main__":
        mp.set_start_method("fork")
        processi = []
        nProc = nMisureSpettro
        manager = mp.Manager()
        tuttiAutovettori = []
        dim = nStati
        for i in range(nProc):
            tuttiAutovettori.append(manager.list())
            for n in range(dim):
                tuttiAutovettori[i].append(manager.list())
            
            pr = mp.Process(target = simQ.trovaTuttiAutostatiMP, args = (HQ, dim, rumore, tuttiAutovettori[i]))
            processi.append(pr)
            pr.start()
        
        for i in range(nProc):
            processi[i].join()
        
        for i in range(nProc):
            autovettori.append([])
            for j in range(dim):
                autovettori[i].append([])
                for k in range(dim):
                    autovettori[i][j].append(tuttiAutovettori[i][j][k])
                    #print(i, j, k, tuttiAutovettori[i][j][k])
    
    #print("autovettori", autovettori)
    return autovettori

condizioniPeriodiche = True
rumore = False # Non implementato

match d:
    case 2:
        H = [[0, 1],
             [1, 0]]
        HQ = SparsePauliOp.from_list([("X", 1)])

    case 4:
        if condizioniPeriodiche:
            H = [[0, -2*x, -2*x, 0],
                [-2*x, 3, 0, -x/2],
                [-2*x, 0, 3, -x/2],
                [0, -x/2, -x/2, 3]]
            c = [9/4, -3/4, -3/4, -3/4, -5*x/4, -3*x/4, -5*x/4, -3*x/4]
        
        else:
            H = [[0, -2*x, -2*x, 0],
                [-2*x, 3, 0, -x],
                [-2*x, 0, 3, -x],
                [0, -x, -x, 4.5]]
            c = [21/8, -9/8, -3/8, -9/8, -3*x/2, -x/2, -3*x/2, -x/2]
            
        HQ = SparsePauliOp.from_list([("II", c[0]), ("ZI", c[1]), ("ZZ", c[2]), ("IZ", c[3]), ("XI", c[4]), ("XZ", c[5]), ("IX", c[6]), ("ZX", c[7])])

    case 8:
        if condizioniPeriodiche:
            cE = [27/8, -3/8, -3/4]
            cB = [-9*x/8, -3*x/8, -x/8]
            HEl = SparsePauliOp.from_list([("III", cE[0]), ("ZZI", cE[1]), ("IZZ", cE[1]), ("ZIZ", cE[1]), ("ZII", cE[2]), ("IZI", cE[2]), ("IIZ", cE[2])])
            HMag = SparsePauliOp.from_list([("XII", cB[0]), ("XIZ", cB[1]), ("XZI", cB[1]), ("XZZ", cB[2]), 
                                            ("IXI", cB[0]), ("ZXI", cB[1]), ("IXZ", cB[1]), ("ZXZ", cB[2]), 
                                            ("IIX", cB[0]), ("IZX", cB[1]), ("ZIX", cB[1]), ("ZZX", cB[2])])
            HQ = SparsePauliOp.sum([HEl, HMag])
            H = HQ.to_matrix()

        else:
            H = [[1, 0, 0, 0, 0, 0, 0, 0],
                [0, 2, 0, 0, 0, 0, 0, 0],
                [0, 0, 3, 0, 0, 0, 0, 0],
                [0, 0, 0, 4, 0, 0, 0, 0],
                [0, 0, 0, 0, 5, 0, 0, 0],
                [0, 0, 0, 0, 0, 6, 0, 0],
                [0, 0, 0, 0, 0, 0, 7, 0],
                [0, 0, 0, 0, 0, 0, 0, 8]]
    case _:
        rng = np.random.default_rng()
        delta = rng.integers(1, d)
        nZeri = rng.integers(0, (d * (d - 1) / 2))
        print("d", d)
        print("delta", delta)
        print("Numero zeri", nZeri)

        matriceQuadrata = rng.random((d, d))
        matriceQuadrata = np.multiply(matriceQuadrata, delta)
        matriceQuadrata = np.add(matriceQuadrata, -(delta / 2))
        for n in range(nZeri):
            i = rng.integers(0, d - 1)
            j = rng.integers(0, d - 1)
            matriceQuadrata[i][j] = 0 # lo stesso elemento potrebbe essere azzerato più volte
        H = np.add(matriceQuadrata, np.transpose(matriceQuadrata))
        print("Hamiltoniana\n", H)

# Risultato analitico
print(H)
autovalori, autovettori = np.linalg.eigh(H)
autovettori = np.transpose(autovettori)

percorsoFile = "../../../data/misureAutovaloriMP/"
if simulazioneQuantistica:
    percorsoFile = "../../../data/misureAutovaloriMPQ/"
nomeFile = percorsoFile + "attesi_" + str(d) + ".txt"
file = open(nomeFile, "w")
for a in autovalori:
    file.write(str(a) + "\n")
file.close()

# Simulazione su circuiti
tI = time.time()
if simulazioneQuantistica:
    tuttiAutostati = misuraSpettroCircuito(nMisureSpettro, HQ, d, rumore)
else:
    tuttiAutostati = misuraSpettroClassica(nMisureSpettro, H)

for esperimento in range(nMisureSpettro):
    autQ = tuttiAutostati[esperimento]

    energie = [moduli.valoreMedio(s, H) for s in autQ]
    errori = [(moduli.varianza(s, H))**0.5 for s in autQ]
    energieOrdinate = energie.copy()
    energieOrdinate.sort()
    erroriOrdinati = []
    autostatiOrdinati = []
    for i in range(len(errori)):
        indice = energie.index(energieOrdinate[i])
        erroriOrdinati.append(errori[indice])
        autostatiOrdinati.append(autQ[indice])
    energie = energieOrdinate
    errori = erroriOrdinati
    autQ = autostatiOrdinati

    print("\nEnergie", energie, autovalori, sep = "\n") 

    for m in range(len(autQ)):
        print("Stato " + str(m), autovettori[m], autQ[m], sep = "\n")

        proiezioni = []
        for n in range(len(autQ)):
            proiezioni.append(abs(np.vdot(autQ[m], autovettori[n])))
        print(proiezioni, "\n")

    nomeFileBase = "Autovalori_d" + str(d)
    if rumore:
        nomeFileBase += "rumore"
    for n in range(d):
        suffissoFile = "_" + str(n) + ".txt"
        nomeFile = percorsoFile + nomeFileBase + suffissoFile
        file = open(nomeFile, "a")
        file.write(str(energie[n]) + "\n")
        file.close()

    nomeFileBase = "Stato_d" + str(d)
    if rumore:
        nomeFileBase += "rumore"
    for n in range(d):
        suffissoFile = "_" + str(n) + ".txt"
        nomeFile = percorsoFile + nomeFileBase + suffissoFile
        file = open(nomeFile, "a")
        file.write(str(autostatiOrdinati[n]) + "\n")
        file.close()

print("Tempo di esecuzione:", time.time() - tI)