import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit.providers.basicaer import QasmSimulatorPy
from qiskit import transpile


def applicaGates(stringaPauli, circuito):
    qr = circuito.qubits
    cr = circuito.clbits

    for i in range(len(stringaPauli)):
        s = stringaPauli[i]

        match s:
            case "I":
                # print("I", i, ": non si fa niente", sep = "")
                continue
            case "X":
                circuito.h(qr[i])
                circuito.measure(qr[i], cr[i])
                # print("X", i, ": si applica H al qubit ", i, " e lo si misura", sep = "")
            case "Y":
                circuito.sdg(qr[i])
                circuito.h(qr[i])
                circuito.measure(qr[i], cr[i])
                # print("Y", i, ": si applica HS+ al qubit ", i, " e lo si misura",  sep = "")
            case "Z":
                circuito.measure(qr[i], cr[i])
                # print("Z", i, ": si misura il qubit ", i, sep = "")
            case _:
                raise Exception("Gate non riconosciuto", s)


def contaUno(stringa):
    nUno = 0
    for c in stringa:
        if c == "1":
            nUno += 1

    return nUno


def stringaPauliValeI(stringa):
    for s in stringa:
        if s != "I":
            return False
    return True


def restringiOperatorePauli(operatore):
    opInLista = operatore.to_list()
    operatoreRistretto = []

    while len(opInLista) > 0:
        stringa = opInLista[0][0]
        # print("Stringa trovata:", stringa)
        coefficiente = 0

        i = 0
        while i < len(opInLista):
            # print(i)
            if stringa == opInLista[i][0]:
                termine = opInLista.pop(i)
                coefficiente += termine[1]
                # print("Trovato un termine:", termine, i)
                continue
            i += 1

        soglia = 0.00001
        if abs(coefficiente) < soglia:
            # print("Stringa scartata:", stringa, coefficiente)
            continue

        operatoreRistretto.append(tuple([stringa, coefficiente]))

    operatoreRistretto = SparsePauliOp.from_list(operatoreRistretto)
    return operatoreRistretto


def valoreAspettazioneStringaPauliDaConteggi(conteggi):
    valore = 0
    for s, c in conteggi.items():
        segno = (-1) ** (contaUno(s))
        valore += segno * c

    return valore


def valoreAspettazioneOp(operatore, circuitoIniziale, backend=None):
    gatePauli = operatore.to_list()
    if backend is None:
        backend = QasmSimulatorPy()
    circuiti = [
        circuitoIniziale.copy(name="circ_" + str(i)) for i in range(len(gatePauli))
    ]

    for i in range(len(gatePauli)):
        stringa = gatePauli[i][0]
        circuito = circuiti[i]
        applicaGates(stringa, circuito)
        if stringaPauliValeI(stringa):
            circuito.clear()
            circuito.measure(0, 0)
        circuitoTradotto = transpile(circuito, backend)
        circuiti[i] = circuitoTradotto

    ripetizioni = 10000  # 100
    res = backend.run(circuiti, shots=ripetizioni).result()

    valoreAspettazione = 0
    for i in range(len(gatePauli)):
        coefficiente = gatePauli[i][1]
        conteggi = res.get_counts(circuiti[i])
        v = valoreAspettazioneStringaPauliDaConteggi(conteggi)
        valoreAspettazione += coefficiente * v / ripetizioni

    return valoreAspettazione


def mediaVarianzaOp(operatore, circuitoIniziale, backend=None):
    operatoreQuadro = operatore.dot(operatore)
    operatoreQuadro = restringiOperatorePauli(operatoreQuadro)
    media = valoreAspettazioneOp(operatore, circuitoIniziale, backend)
    varianza = valoreAspettazioneOp(operatoreQuadro, circuitoIniziale, backend)
    varianza = varianza - (media**2)
    return [np.real(media), np.real(varianza)]


# # 100 ripetizioni in circa 1 minuto
# tI = time.time()

# rip = 100
# valori = []
# varianze = []

# x = 1.5
# H = [[0, -2*x, -2*x, 0],
#     [-2*x, 3, 0, -x/2],
#     [-2*x, 0, 3, -x/2],
#     [0, -x/2, -x/2, 3]]
# c = [9/4, -3/4, -3/4, -3/4, -5*x/4, -3*x/4, -5*x/4, -3*x/4]
# operatore = SparsePauliOp.from_list([("II", c[0]), ("ZI", c[1]), ("ZZ", c[2]), ("IZ", c[3]), ("XI", c[4]), ("XZ", c[5]), ("IX", c[6]), ("ZX", c[7])])
# #print(operatore.power(2).to_list())

# stati = [[0.80659023, 0.41172526, 0.41172526, 0.10186665],
#          [-0.23362733,  0.11011279,  0.11011279,  0.9597753],
#          [ 0.,          0.70710678, -0.70710678,  0.        ],
#          [-0.54298294,  0.56423176,  0.56423176, -0.26163824]]
# altriStati = []

# for i in range(rip):
#     # rng = np.random.default_rng()
#     # stato = rng.random(size = 4)
#     # stato = [(p - 0.5) for p in stato]
#     stato = [-0.15675171531120913, 0.4245819024436823, -0.4237918889884159, 0.3872576967944421]
#     stato = np.multiply(stato, 1 / np.linalg.norm(stato))
#     altriStati.append(stato)

#     qr = QuantumRegister(2)
#     cr = ClassicalRegister(2)
#     qc = QuantumCircuit(qr, cr)
#     qc.prepare_state(stato, qr)
#     v, v2 = mediaVarianzaOp(operatore, qc)
#     valori.append(v)
#     varianze.append(v2)

# conteggi, bins = np.histogram(valori, 20, (min(valori), max(valori)))
# plt.figure(figsize = (10, 5))
# plt.stairs(conteggi, bins, label = "Misure")
# plt.savefig("spettroGauge/istoMisure.png")
# plt.close()

# print("Atteso", simCl.valoreMedio(stato, H), simCl.varianza(stato, H), stato)
# for i in range(len(valori)):
#     print(valori[i], varianze[i], altriStati[i])
# #print(valori, varianze, sep = "\n")
# print(np.average(valori), np.average(varianze))

# print(rip, "ripetizioni in", time.time() - tI)

# # [-0.43374384400571886, 0.2063100948198633, 0.36543621706369245, 0.4626637778456697]
# # [-0.15675171531120913, 0.4245819024436823, -0.4237918889884159, 0.3872576967944421]
