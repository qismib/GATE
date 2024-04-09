from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
# Il transpiler non supporta la nuova versione dei backend (BackendV2), quindi bisogna usare quelli vecchi
from qiskit.providers.fake_provider import FakeLagos
from qiskit.primitives import BackendSampler
from qiskit.providers.basicaer import QasmSimulatorPy
from qiskit.compiler import transpile
from qiskit.transpiler import Layout, TranspileLayout
import numpy as np
import matplotlib.pyplot as plt
import math

#### Funzioni utili

def dizionarioInLista(res, statiAttesi):
    lista = [0 for s in statiAttesi]

    for stato, misura in res.items():
        if stato not in statiAttesi:
            raise Exception("dizionarioInLista():", stato, "non è presente fra gli stati attesi", statiAttesi)
        indice = statiAttesi.index(stato)
        lista[indice] = misura

    return lista

## test dizionarioInLista
# statiAtt = ["00", "01", "10", "11"]
# diz = {"01" : 5, "10" : 3, "00" : -1, "11" : 8}
# print(dizionarioInLista(diz, statiAtt))

def preparaStatoBaseComp(circuito, stato):
    nQubit = len(stato)
    if nQubit > circuito.num_qubits:
        raise Exception("Non è possibile generare lo stato", stato, "perchè il circuito ha solo", circuito.num_qubits, "qubit")
    
    for j in range(len(stato)):
        bit = stato[j]
        i = len(stato) - j - 1
        if bit == "1":
            circuito.x(circuito.qubits[i])
            #print("X sul qubit", i)
        elif bit != "0":
            raise Exception(stato, "non è uno stato valido da generare")

## test preparaStatoBaseComp
# qc = QuantumCircuit(4, 4)
# preparaStatoBaseComp(qc, "0101")

# misure[k][n]:
# k: indice stato
# n: indice temporale
def scriviMisureSuFile(misure, prefissoFile):
    nStati = len(misure)
    nPunti = len(misure[0])

    for k in range(nStati):
        nomeFile = prefissoFile + str(k) + ".txt"
        file = open(nomeFile, "w")
        for n in range(nPunti):
            file.write(str(misure[k][n]))
            file.write("\n")
        file.close()


def leggiMisureDaFile(prefissoFile, nStati, nPunti):
    misure = [[] for k in range(nStati)]

    for k in range(nStati):
        nomeFile = prefissoFile + str(k) + ".txt"
        file = open(nomeFile, "r")
        for n in range(nPunti):
            s = file.readline()
            if s == "":
                #print("Letta una linea vuota")
                continue
            mis = float(s)
            misure[k].append(mis)
        file.close()
    
    return misure


def sovrapposizione(stati, coefficienti):
    v = [0 for n in range(len(stati))]
    for n in range(len(stati)):
        v = np.add(v, np.multiply(stati[n], coefficienti[n]))
    
    return v


#### Ricalibrazione delle misure

def matriceCalibrazioneErroriMisura(circuito, backend_sim, sampler, statiAttesi, ripetizioni = 10000):
    nQubit = circuito.num_qubits
    nStati = 2**nQubit
    if nStati != len(statiAttesi):
        raise Exception("Il numero di stati attesi (", len(statiAttesi),") non coincide col numero di qubit presenti nel circuito (", nStati, ")", sep = "")
    matriceCalibrazione = [[0 for m in range(nStati)] for n in range(nStati)]

    for n in range(nStati):
        preparaStatoBaseComp(circuito, statiAttesi[n])
        circuito.measure_all(add_bits = False)

        circuitoTradotto = transpile(circuito, backend = backend_sim)
        res = sampler.run(circuits = [circuitoTradotto], shots = ripetizioni).result().quasi_dists[0].binary_probabilities()
        for stato, misura in res.items():
            indice = statiAttesi.index(stato)
            matriceCalibrazione[indice][n] = misura

        circuito.clear()
    
    matriceCalibrazione = np.linalg.inv(matriceCalibrazione)
    return matriceCalibrazione

#### Test calibrazione
# nQubit = 2
# qr = QuantumRegister(nQubit, name = "Ciao")
# cr = ClassicalRegister(nQubit)
# qc = QuantumCircuit(qr, cr)
# backend_sim = FakeLagos()
# sampler = BackendSampler(backend_sim)

# statiAtt = [f"{n:0{nQubit}b}" for n in range(2**nQubit)]
# cal = matriceCalibrazioneErroriMisura(qc, backend_sim, sampler, statiAtt)
# print("Matrice calibrazione:\n", cal)

# for s in statiAtt:
#     preparaStatoBaseComp(qc, s)
#     qc.measure_all(add_bits = False)
#     qc.draw(output = "mpl", filename = "EvoluzionePlacchette/circ" + s + ".png")
#     circuitoTradotto = transpile(qc, backend = backend_sim)
#     res = sampler.run(circuits = [circuitoTradotto], shots = 10000).result().quasi_dists[0].binary_probabilities()
#     statoMis = dizionarioInLista(res, statiAtt)

#     print(s, "\n", statoMis, sum(statoMis), np.matmul(cal, statoMis), sum(np.matmul(cal, statoMis)))

#     qc.clear()
####

# for k in range(nQubit):
#     # Si prepara lo stato con k = 0 e si misura solo k
#     matriceMisure[k].append([])
#     qc.measure(qr[k], cr[0])
#     circuitoTradotto = transpile(qc, backend = backend_sim)
#     #circuitoTradotto.draw(output = "mpl", filename = "circCal0" + str(k) + ".png")
#     res = sampler.run(circuits = [circuitoTradotto], shots = ripetizioni).result().quasi_dists[0].binary_probabilities()
#     matriceMisure[k][0].append(res.get("00")) # Queste stringhe dipendono dal numero di qubit
#     matriceMisure[k][0].append(res.get("01"))
#     qc.clear()

#     # Si prepara lo stato con k = 1 e si misura solo k
#     matriceMisure[k].append([])
#     qc.x(qr[k])
#     qc.measure(qr[k], cr[0])
#     circuitoTradotto = transpile(qc, backend = backend_sim)
#     #circuitoTradotto.draw(output = "mpl", filename = "circCal1" + str(k) + ".png")
#     res = sampler.run(circuits = [circuitoTradotto], shots = ripetizioni).result().quasi_dists[0].binary_probabilities()
#     matriceMisure[k][1].append(res.get("00")) # Queste stringhe dipendono dal numero di qubit
#     matriceMisure[k][1].append(res.get("01"))
#     qc.clear()


def correggiErroreMisura(misure, mCal):
    return np.matmul(mCal, misure)


#### Randomized compiling

def cxCasuale(circuito, qubitControllo, qubitBersaglio):
    n = np.random.randint(16)
    
    match n:
        case 0:
            circuito.cx(qubitControllo, qubitBersaglio)
        case 1:
            circuito.x(qubitControllo)
            circuito.cx(qubitControllo, qubitBersaglio)
            circuito.x(qubitControllo)
            circuito.x(qubitBersaglio)
        case 2:
            circuito.y(qubitControllo)
            circuito.cx(qubitControllo, qubitBersaglio)
            circuito.y(qubitControllo)
            circuito.x(qubitBersaglio)
        case 3:
            circuito.z(qubitControllo)
            circuito.cx(qubitControllo, qubitBersaglio)
            circuito.z(qubitControllo)
        case 4:
            circuito.x(qubitBersaglio)
            circuito.cx(qubitControllo, qubitBersaglio)
            circuito.x(qubitBersaglio)
        case 5:
            circuito.x(qubitControllo)
            circuito.x(qubitBersaglio)
            circuito.cx(qubitControllo, qubitBersaglio)
            circuito.x(qubitControllo)
        case 6:
            circuito.y(qubitControllo)
            circuito.x(qubitBersaglio)
            circuito.cx(qubitControllo, qubitBersaglio)
            circuito.y(qubitControllo)
        case 7:
            circuito.z(qubitControllo)
            circuito.x(qubitBersaglio)
            circuito.cx(qubitControllo, qubitBersaglio)
            circuito.z(qubitControllo)
            circuito.x(qubitBersaglio)
        case 8:
            circuito.y(qubitBersaglio)
            circuito.cx(qubitControllo, qubitBersaglio)
            circuito.z(qubitControllo)
            circuito.y(qubitBersaglio)
        case 9:
            circuito.x(qubitControllo)
            circuito.y(qubitBersaglio)
            circuito.cx(qubitControllo, qubitBersaglio)
            circuito.y(qubitControllo)
            circuito.z(qubitBersaglio)
        case 10:
            circuito.y(qubitControllo)
            circuito.y(qubitBersaglio)
            circuito.cx(qubitControllo, qubitBersaglio)
            circuito.x(qubitControllo)
            circuito.z(qubitBersaglio)
        case 11:
            circuito.z(qubitControllo)
            circuito.y(qubitBersaglio)
            circuito.cx(qubitControllo, qubitBersaglio)
            circuito.y(qubitBersaglio)
        case 12:
            circuito.z(qubitBersaglio)
            circuito.cx(qubitControllo, qubitBersaglio)
            circuito.z(qubitControllo)
            circuito.z(qubitBersaglio)
        case 13:
            circuito.x(qubitControllo)
            circuito.z(qubitBersaglio)
            circuito.cx(qubitControllo, qubitBersaglio)
            circuito.y(qubitControllo)
            circuito.y(qubitBersaglio)
        case 14:
            circuito.y(qubitControllo)
            circuito.z(qubitBersaglio)
            circuito.cx(qubitControllo, qubitBersaglio)
            circuito.x(qubitControllo)
            circuito.y(qubitBersaglio)
        case 15:
            circuito.z(qubitControllo)
            circuito.z(qubitBersaglio)
            circuito.cx(qubitControllo, qubitBersaglio)
            circuito.z(qubitBersaglio)
        case _:
            print("Errore!")
    
    return n

#Ridefinizione dei cnot per zero-noise-extrapolation
def gruppoCx(circuito, qubitControllo, qubitBersaglio, numeroCx):
    for n in range(numeroCx):
        cxCasuale(circuito, qubitControllo, qubitBersaglio)

# qr = QuantumRegister(2)
# cr = ClassicalRegister(2)
# qc = QuantumCircuit(qr, cr)

# for n in range(100):
#     gate = cnotCasuale(qc, qr[0], qr[1])
#     qc.draw(output = "mpl", filename = "Randomized" + str(gate) + ".png")
#     qc.clear()


#### Evoluzione temporale su hardware quantistico

# def passoOtt(circuito, qr1, qr2, t, x, nCx):
#     par = [-(x * t / 2), 
#            -(3 * t / 8), 
#            -(3 * x * t / 2), 
#            -(9 * t / 4), 
#            -(9 * t / 8), 
#            -(3 * x * t), 
#            -(x * t)]

#     circuito.ry(par[0], qr1)
#     circuito.rz(par[1], qr1)
#     gruppoCx(circuito, qr2, qr1, nCx)

#     circuito.rz(par[4], qr2)
#     circuito.ry(par[2], qr1)

#     gruppoCx(circuito, qr1, qr2, nCx)
#     circuito.ry(par[6], qr2)
#     gruppoCx(circuito, qr1, qr2, nCx)

#     circuito.ry(par[5], qr2)
#     circuito.rz(par[4], qr2)
#     circuito.rz(par[3], qr1)
#     circuito.ry(par[2], qr1)

#     gruppoCx(circuito, qr2, qr1, nCx)
#     circuito.rz(par[1], qr1)
#     circuito.ry(par[0], qr1)


def ciclo2Placchette(circuito, qr, t, x, nCx, c, d):
    qr1 = qr[0]
    qr2 = qr[1]
    par = [-(3 * x * t / 2), 
           -(3 * t / 2), 
           -(5 * x * t / 2), 
           -(3 * t / 2)]

    circuito.ry(d * par[0], qr1)
    circuito.rz(d * par[1], qr1)
    gruppoCx(circuito, qr2, qr1, nCx)

    circuito.ry(d * par[2], qr1)
    circuito.rz(d * par[3], qr2)

    gruppoCx(circuito, qr1, qr2, nCx)
    circuito.ry((c + d) * par[0], qr2)
    gruppoCx(circuito, qr1, qr2, nCx)

    circuito.ry((c + d) * par[2], qr2)
    circuito.rz((c + d) * par[3], qr1)
    circuito.rz(c * par[3], qr2)
    circuito.ry(c * par[2], qr1)

    gruppoCx(circuito, qr2, qr1, nCx)
    circuito.rz(c * par[1], qr1)
    circuito.ry(c * par[0], qr1)


def ciclo3Placchette(circuito, qr, t, x, nCx, c, d):
    par = [-(3 * t / 4), 
           -(3 * t / 2), 
           -(9 * x * t / 4), 
           -(3 * x * t / 4),
           -(x * t / 4)]

    vi = [0, 1, 2]
    vj = [1, 2, 0]
    vk = [2, 0, 1]
    for n in range(len(vi)):
        circuito.rz(d * par[1], qr[vi[n]]) ###
        circuito.ry(d * par[2], qr[vi[n]])

        gruppoCx(circuito, qr[vi[n]], qr[vj[n]], nCx)
        circuito.rz(d * par[0], qr[vj[n]])
        circuito.ry(d * par[3], qr[vj[n]])
        gruppoCx(circuito, qr[vk[n]], qr[vj[n]], nCx)
        circuito.ry(d * par[4], qr[vj[n]])
        gruppoCx(circuito, qr[vi[n]], qr[vj[n]], nCx)
        circuito.ry(d * par[3], qr[vj[n]])
        gruppoCx(circuito, qr[vk[n]], qr[vj[n]], nCx)
    
    for n in range(len(vi) - 1, -1, -1):
        gruppoCx(circuito, qr[vk[n]], qr[vj[n]], nCx)
        circuito.ry(c * par[3], qr[vj[n]])
        gruppoCx(circuito, qr[vi[n]], qr[vj[n]], nCx)
        circuito.ry(c * par[4], qr[vj[n]])
        gruppoCx(circuito, qr[vk[n]], qr[vj[n]], nCx)
        circuito.ry(c * par[3], qr[vj[n]])
        circuito.rz(c * par[0], qr[vj[n]])
        gruppoCx(circuito, qr[vi[n]], qr[vj[n]], nCx)

        circuito.ry(c * par[2], qr[vi[n]])
        circuito.rz(c * par[1], qr[vi[n]])


def passoGenerico(circuito, qr, t, x, nCx, a, b):
    if len(a) != len(b):
        raise Exception("passoGenerico(): i coefficienti passati non corrispondono", a, b)
    
    c = [a[0]]
    d = [b[0] - a[0]]
    for n in range(1, len(a)):
        c.append(a[n] - d[n - 1])
        d.append(b[n] - c[n])

    nCicli = len(a)
    nPlacchette = len(qr)
    if nPlacchette == 2:
        for n in range(nCicli - 1, -1, -1):
            ciclo2Placchette(circuito, qr, t, x, nCx, c[n], d[n])
    elif nPlacchette == 3:
        for n in range(nCicli - 1, -1, -1):
            ciclo3Placchette(circuito, qr, t, x, nCx, c[n], d[n])


def passoVerlet(circuito, qr, t, x, nCx):
    a = [0.5]
    b = [1]
    passoGenerico(circuito, qr, t, x, nCx, a, b)


def passoBlanesMoan(circuito, qr, t, x, nCx):
    a = [0.07920369643119569, 0.353172906049774, -0.0420650803577195]
    b = [0.209515106613362, -0.143851773179818]

    a3 = 1 - (2 * sum(a))
    a.append(a3)
    # a(q - i) = a(i), q = 6
    a.append(a[2]) # a4
    a.append(a[1]) # a5

    b2 = 0.5 - sum(b)
    b.append(b2)
    # b(q - 1 - i) = b(i)
    b.append(b[2]) # b3
    b.append(b[1]) # b4
    b.append(b[0]) # b5

    passoGenerico(circuito, qr, t, x, nCx, a, b)


def passoSuzuki(circuito, qr, t, x, nCx):
    a = [0.2072453858971879, 0.4144907717943757]
    b = [0.4144907717943757, 0.4144907717943757]

    a2 = 0.5 - sum(a)
    a.append(a2)
    a.append(a[2]) # a3
    a.append(a[1]) # a4

    b2 = 1 - (2 * sum(b))
    b.append(b2)
    b.append(b[1]) # b3
    b.append(b[0]) # b4

    passoGenerico(circuito, qr, t, x, nCx, a, b)
    

def evoluzione2Pl(circuito, registro, dt, x, passi, nCx, tipo = "Verlet"):
    if passi == 0:
        return

    circuito.s(registro)
    gruppoCx(circuito, registro[1], registro[0], nCx)
    for k in range(passi):
        match tipo:
            case "Suzuki":
                passoSuzuki(circuito, registro, dt, x, nCx)
            case "BM":
                passoBlanesMoan(circuito, registro, dt, x, nCx)
            case "Verlet":
                passoVerlet(circuito, registro, dt, x, nCx)
            case _:
                raise Exception("Decomposizione", tipo, "non nota")
    gruppoCx(circuito, registro[1], registro[0], nCx)
    circuito.sdg(registro)


def evoluzione3Pl(circuito, registro, dt, x, passi, nCx, tipo = "Verlet"):
    if passi == 0:
        return

    circuito.s(registro)
    for k in range(passi):
        match tipo:
            case "Suzuki":
                passoSuzuki(circuito, registro, dt, x, nCx)
            case "BM":
                passoBlanesMoan(circuito, registro, dt, x, nCx)
            case "Verlet":
                passoVerlet(circuito, registro, dt, x, nCx)
            case _:
                raise Exception("Decomposizione", tipo, "non nota")
    circuito.sdg(registro)


def evoluzionePerAutomitigazione(circuito, registro, dt, x, passiAvanti, nCx):
    if passiAvanti == 0:
        return

    circuito.s(registro)
    gruppoCx(circuito, registro[1], registro[0], nCx)
    for k in range(passiAvanti):
        passoBlanesMoan(circuito, registro,  dt, x, nCx)
        #passoVerlet(circuito, registro,  dt, x, nCx)
    for k in range(passiAvanti):
        passoBlanesMoan(circuito, registro, -dt, x, nCx)
        #passoVerlet(circuito, registro, -dt, x, nCx)
    gruppoCx(circuito, registro[1], registro[0], nCx)
    circuito.sdg(registro)


#### Evoluzione hamiltoniana esatta

def evoluzioneHamiltoniana(statoIniziale, autovalori, matriceAutovettori, matriceAutovettoriInversa, t):
    dim = len(autovalori)

    matriceEvDiagonale = np.eye(N = dim, dtype = np.cdouble)
    for n in range(dim):
        matriceEvDiagonale[n][n] = matriceEvDiagonale[n][n] * np.e**(-1j * autovalori[n] * t)
    
    statoEvoluto = np.matmul(matriceAutovettori, matriceEvDiagonale)
    statoEvoluto = np.matmul(statoEvoluto, matriceAutovettoriInversa)
    statoEvoluto = np.matmul(statoEvoluto, statoIniziale)
    return statoEvoluto


def evoluzioneTemporaleStato(statoIniziale, H, tempi):
    nStati = len(statoIniziale)
    nPunti = len(tempi)
    vettoriStato = []

    autovalori, matriceAutovettori = np.linalg.eigh(H)
    matriceAutovettoriInversa = np.linalg.inv(matriceAutovettori)
    for t in tempi:
        stato = evoluzioneHamiltoniana(statoIniziale, autovalori, matriceAutovettori, matriceAutovettoriInversa, t)
        vettoriStato.append(stato)

    return vettoriStato

#### Test evoluzione hamiltoniana
# statoIniziale = [0, 1, 0, 0]
# x = 1.5
# H = [[0, -2*x, -2*x, 0],
#      [-2*x, 3, 0, -x],
#      [-2*x, 0, 3, -x],
#      [0, -x, -x, 4.5]]
# autovalori, matriceAutovettori = np.linalg.eigh(H)
# matriceAutovettoriInversa = np.linalg.inv(matriceAutovettori)

# nPunti = 1000
# tMax = 5
# ascisse = np.linspace(0, tMax, nPunti)
# prob = [[] for n in range(len(statoIniziale))]
# partiReali = [[] for n in range(len(statoIniziale))]
# partiImmaginarie = [[] for n in range(len(statoIniziale))]

# for t in ascisse:
#     stato = evoluzioneHamiltoniana(statoIniziale, autovalori, matriceAutovettori, matriceAutovettoriInversa, t)
#     stato = eliminaFaseGlobale(stato, 0)
    
#     for m in range(len(statoIniziale)):
#         prob[m].append(abs(stato[m]) ** 2)
#         partiReali[m].append(np.real(stato[m]))
#         partiImmaginarie[m].append(np.imag(stato[m]))

# creaImmagine([ascisse], [prob], "Prob")
# creaImmagine([ascisse], [partiReali], "PR")
# creaImmagine([ascisse], [partiImmaginarie], "PI")
####

#### Zero Noise Extrapolation

def ZNE(misure):
    itZNE = len(misure)
    if itZNE == 1:
        return misure[0]
    
    ascisse =  [((2 * m) + 1) for m in range(itZNE)]
    ordinate = misure
    retta = np.polynomial.Polynomial.fit(ascisse, ordinate, deg = 1)
    misuraZNE = retta.convert().coef[0]
    return misuraZNE

# vettore[k][n][i]:
# k: indice stato
# n: indice temporale
# i: indice ZNE
def ZNEMisureAccorpate(vettore):
    nStati = len(vettore)
    nPunti = len(vettore[0])
    res = [[] for k in range(nStati)]

    for k in range(nStati):
        for n in range(nPunti):
            mis = ZNE(vettore[k][n])
            res[k].append(mis)
    
    return res

#### Test ZN
# itZN = 3
# misuraSingola = [8]
# misure = [5, 6, 8]
# misureAcc = [[[1, 3], [2, 4], [3, 5]], [[0, -1], [-2, -3.3], [4, 1]]]

# print(ZNE(misuraSingola), ZNE(misure))
# print(ZNEMisureAccorpate(misureAcc))
####

#### Auto mitigazione

def automitigazione(misuraFisica, misuraMitigante, probAttesa, nStati):
    rumore = 1 / nStati
    soglia = 0.01
    mF = misuraFisica - rumore
    mM = misuraMitigante - rumore
    pA = probAttesa - rumore

    probFisicaVera = rumore + (mF * pA / mM)

    if probFisicaVera < 0 and probFisicaVera > -soglia:
        print("automitigazione(): la probabilità calcolata è negativa sotto la soglia, e viene corretta", probFisicaVera)
        probFisicaVera = 0
    elif probFisicaVera < -soglia:
        print("automitigazione(): la probabilità calcolata è negativa oltre la soglia e viene scartata", probFisicaVera)
        return 0
    elif probFisicaVera > 1 + soglia:
        print("automitigazione(): la probabilità calcolata è maggiore di uno e viene scartata", probFisicaVera)
        return 0
    if abs(mM) < soglia:
        print("automitigazione(): la misura mitigante è troppo vicina al rumore: automitigazione da scartare", probFisicaVera)
        return 0

    return probFisicaVera

# vettoreMis[k][n], vettoreAuto[k][nMit], probAttese[k]
# k: indice stato
# n: indice temporale
# nMit: indice temporale con metà dei punti
# le misure mitigate sono la metà di quelle fisiche
def automitigazioneMisureAccorpate(vettoreMis, vettoreAuto, probAttese):
    nStati = len(vettoreMis)
    nPunti = len(vettoreMis[0])
    misureMitigate = [[] for k in range(nStati)]

    for n in range(0, nPunti, 2):
        nMit = int(n / 2)
        for k in range(nStati):
            probMisurataFisica = vettoreMis[k][n]
            probMisurataMitigazione = vettoreAuto[k][nMit]
            probVeraMitigazione = probAttese[k]

            misuraMit = automitigazione(probMisurataFisica, probMisurataMitigazione, probVeraMitigazione, nStati)
            misureMitigate[k].append(misuraMit)
    
    return misureMitigate