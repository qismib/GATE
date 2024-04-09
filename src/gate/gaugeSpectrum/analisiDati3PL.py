import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from valoreAspettazionePauli import valoreAspettazioneOp
from qiskit.quantum_info import SparsePauliOp
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

DATA_PATH = "../../../data"

d = 8
x = 1.5
leggiFile = True
metodi = ["VVQE"]


def medieRaggruppate(dati, dim=30):
    N = int(len(dati) / dim)
    medie = [[] for n in range(N)]
    errori = []
    for n in range(len(dati)):
        i = int(n / dim)
        medie[i].append(dati[n])

    for n in range(len(medie)):
        errori.append(np.std(medie[n]) / len(medie[n]) ** 0.5)
        medie[n] = np.mean(medie[n])

    return medie, errori


def misuraEnergiaSuCircuito(stato, H, ripetizioni=30):
    nQubit = int(np.log2(len(stato)))
    qr = QuantumRegister(nQubit)
    cr = ClassicalRegister(nQubit)
    qc = QuantumCircuit(qr, cr)
    qc.prepare_state(stato, normalize=True)

    valori = []
    for i in range(ripetizioni):
        v = valoreAspettazioneOp(H, qc)
        valori.append(v)

    return np.real(valori)


def scriviMisureSuFile(misure, nomeFile):
    file = open(nomeFile, "w")
    for m in misure:
        file.write(str(m) + "\n")
    file.close()


def leggiMisureDaFile(nomeFile):
    file = open(nomeFile, "r")
    stringheMisure = file.read().split("\n")
    stringheMisure.pop()
    misure = [float(s) for s in stringheMisure]
    file.close()
    return misure


def leggiVettoriDaFile(nomeFile):
    file = open(nomeFile, "r")
    stringheVettori = file.read().split("\n")
    stringheVettori.pop()
    vettori = []
    for i in range(len(stringheVettori)):
        vettore = [float(s) for s in stringheVettori[i][1:-2].split(",")]
        vettori.append(vettore)
    file.close()
    print(vettori)
    return vettori


multiprocessing = True

if d == 4:
    c = [9 / 4, -3 / 4, -3 / 4, -3 / 4, -5 * x / 4, -3 * x / 4, -5 * x / 4, -3 * x / 4]
    HQ = SparsePauliOp.from_list(
        [
            ("II", c[0]),
            ("ZI", c[1]),
            ("ZZ", c[2]),
            ("IZ", c[3]),
            ("XI", c[4]),
            ("XZ", c[5]),
            ("IX", c[6]),
            ("ZX", c[7]),
        ]
    )

elif d == 8:
    cE = [27 / 8, -3 / 8, -3 / 4]
    cB = [-9 * x / 8, -3 * x / 8, -x / 8]
    HEl = SparsePauliOp.from_list(
        [
            ("III", cE[0]),
            ("ZZI", cE[1]),
            ("IZZ", cE[1]),
            ("ZIZ", cE[1]),
            ("ZII", cE[2]),
            ("IZI", cE[2]),
            ("IIZ", cE[2]),
        ]
    )
    HMag = SparsePauliOp.from_list(
        [
            ("XII", cB[0]),
            ("XIZ", cB[1]),
            ("XZI", cB[1]),
            ("XZZ", cB[2]),
            ("IXI", cB[0]),
            ("ZXI", cB[1]),
            ("IXZ", cB[1]),
            ("ZXZ", cB[2]),
            ("IIX", cB[0]),
            ("IZX", cB[1]),
            ("ZIX", cB[1]),
            ("ZZX", cB[2]),
        ]
    )
    HQ = SparsePauliOp.sum([HEl, HMag])

H = HQ.to_matrix()
autovalori, autovettori = np.linalg.eigh(H)
autovettori = np.transpose(autovettori)

percorsoFile = f"{DATA_PATH}/misureAutovaloriMP/"
if multiprocessing:
    percorsoFile = f"{DATA_PATH}/misureAutovaloriMPQ/"

print(d)
rng = np.random.default_rng()

for esperimento in range(d):
    erroriSovrMetodi = []
    for metodo in range(len(metodi)):
        met = metodi[metodo]

        suffissoFile = str(d) + "_" + str(esperimento) + "_" + met
        nomeFile = percorsoFile + "Stato_d" + suffissoFile + ".txt"
        autovettoriStimati = leggiVettoriDaFile(nomeFile)
        nMisCampione = len(autovettoriStimati)
        print("Autovalore:", esperimento, nMisCampione)

        nMisureMedia = 30
        if not leggiFile:
            tutteMedie = []
            medie = []
            errori = []
            for i in range(nMisCampione):
                energieMisurate = misuraEnergiaSuCircuito(
                    autovettoriStimati[i], HQ, nMisureMedia
                )
                print(energieMisurate)
                tutteMedie = tutteMedie + energieMisurate.tolist()
                medie.append(np.mean(energieMisurate))
                errori.append(
                    np.std(energieMisurate) / (len(energieMisurate) - 1) ** 0.5
                )

            nomeFile = percorsoFile + "medie_" + suffissoFile + ".txt"
            scriviMisureSuFile(medie, nomeFile)
            nomeFile = percorsoFile + "errori_" + suffissoFile + ".txt"
            scriviMisureSuFile(errori, nomeFile)

        else:
            nomeFile = percorsoFile + "medie_" + suffissoFile + ".txt"
            medie = leggiMisureDaFile(nomeFile)
            nomeFile = percorsoFile + "errori_" + suffissoFile + ".txt"
            errori = leggiMisureDaFile(nomeFile)

        mediaMisurata = np.mean(medie)
        erroreMisurato = np.std(medie) / (len(medie) - 1) ** 0.5
        distanzaDevSt = abs(mediaMisurata - autovalori[esperimento]) / erroreMisurato
        biasPercentuale = (
            (mediaMisurata - autovalori[esperimento]) / autovalori[esperimento] * 100
        )

        erroriSovr = []
        for i in range(nMisCampione):
            errSov = (
                1 - abs(np.vdot(autovettori[esperimento], autovettoriStimati[i])) ** 2
            )
            if esperimento == 2 or esperimento == 6:  # degeneri con quello prima
                errSov -= (
                    abs(np.vdot(autovettori[esperimento - 1], autovettoriStimati[i]))
                    ** 2
                )
            elif esperimento == 1 or esperimento == 5:  # degeneri con quello dopo
                errSov -= (
                    abs(np.vdot(autovettori[esperimento + 1], autovettoriStimati[i]))
                    ** 2
                )

            erroriSovr.append(errSov)
        erroriSovrMetodi.append(erroriSovr)

        print("Valore atteso:", autovalori[esperimento])
        print("Media misurata:", mediaMisurata)
        print("Deviazione standard:", erroreMisurato)
        print("Distanza in deviazioni standard:", distanzaDevSt)
        print("Bias percentuale: ", biasPercentuale, "%", sep="")
        print("Errore sovrapposizione:", erroriSovr)
        print("\n")

        file = open(percorsoFile + "esito_" + suffissoFile + ".txt", "w")
        file.write("Valore atteso: " + str(autovalori[esperimento]) + "\n")
        file.write("Esecuzioni circuiti:" + str(nMisureMedia) + "\n")
        file.write("Media misurata: " + str(mediaMisurata) + "\n")
        file.write("Deviazione standard: " + str(erroreMisurato) + "\n")
        file.write("Distanza in deviazioni standard: " + str(distanzaDevSt) + "\n")
        file.write("Bias percentuale: " + str(biasPercentuale) + "\n")
        file.write("Sovrapposizioni: " + str(erroriSovr) + "\n")

        font = {"family": "serif", "weight": "normal", "size": 40}

        matplotlib.rc("font", **font)

        indiciMisure = [(n + 1) for n in range(len(medie))]
        plt.figure(figsize=(15, 10))
        plt.subplots_adjust(top=0.98, bottom=0.18, left=0.08, right=0.95)

        # plt.title("Energie misurate dello stato " + str(esperimento))
        plt.xlabel("Energy in units of $g^2 / 2$", labelpad=30)
        plt.ylabel("Measurements", labelpad=20)
        plt.yticks(ticks=[])
        plt.errorbar(
            medie,
            indiciMisure,
            xerr=errori,
            fmt="o",
            label="Misure",
            markersize=10,
            linewidth=2,
        )
        plt.errorbar(
            [mediaMisurata],
            [len(medie) + 1],
            xerr=[erroreMisurato],
            fmt="^",
            label="Stima del valore atteso",
            markersize=10,
            linewidth=2,
        )
        plt.vlines(
            autovalori[esperimento],
            ymin=0,
            ymax=len(medie) + 2,
            colors=["r"],
            label="Valore atteso",
            linewidths=3,
        )
        # plt.legend()
        plt.savefig(percorsoFile + "medieConErrore_" + suffissoFile + ".png")
        plt.close()

    figura, asse = plt.subplots(figsize=(15, 10))
    if esperimento == 0:
        asse.ticklabel_format(style="sci", scilimits=(0, 0))
    plt.subplots_adjust(top=0.98, bottom=0.14, left=0.12, right=0.92)

    # plt.xlabel("Error", labelpad = 10)
    # plt.ylabel("Errore sovrapposizione")

    asse.barh(
        indiciMisure, erroriSovrMetodi[0], align="center", color="b", label="VVQE"
    )

    # plt.legend()
    asse.set_yticks([3], labels=["VVQE"])
    plt.savefig(percorsoFile + "ErroreSovr_" + suffissoFile + ".png")
    plt.close()
