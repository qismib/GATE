import modulesEvolution as moduli
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

DATA_PATH = "../../../data"

nPlacchette = 3  # 2, 3
T = 40
nPunti = 400
misuraUnQubit = True

nStati = 4
attesi = [-3.06270952, 2.82790849, 3.0, 6.23480103]
percorsoFile = f"{DATA_PATH}/frequencies2PL/"
prefissoMisure = "prob2P_"

if nPlacchette == 3:
    nStati = 8
    attesi = [
        -4.3095234430356815,
        2.0729490168751568,
        2.072949016875158,
        2.694114862206689,
        5.025139089404018,
        5.427050983124841,
        5.427050983124841,
        8.590269491424971,
    ]
    percorsoFile = f"{DATA_PATH}/frequencies3PL/"
    prefissoMisure = "prob3P_"

gapAttesi = []
gapMisurati = []
gapErroriMin = []
gapErroriMax = []

for s in range(1, nStati):
    fileMisure = percorsoFile + prefissoMisure + str(s)
    probStati = moduli.leggiMisureDaFile(fileMisure, nStati, nPunti)

    if misuraUnQubit:
        nMisUnQubit = 2
        probUnQubit = [[] for _ in range(nMisUnQubit)]

        for n in range(nPunti):
            probQubit0 = 0
            probQubit1 = 0
            for k in range(0, nStati, 2):
                probQubit0 += probStati[k][n]
                probQubit1 += probStati[k + 1][n]

            probUnQubit[0].append(probQubit0)
            probUnQubit[1].append(probQubit1)

        probStati = probUnQubit

    nMisurePerSpettro = len(probStati)
    ascisse = [(n + 1) for n in range(nMisurePerSpettro)]

    #### Analisi delle frequenze
    trFourier = [[] for _ in range(nMisurePerSpettro)]
    for m in range(len(trFourier)):
        # trFourier[m] = np.fft.rfft(ordinateProb[m])
        trFourier[m] = np.fft.rfft(probStati[m])
        trFourier[m][0] = 0
        for n in range(len(trFourier[m])):
            trFourier[m][n] = abs(trFourier[m][n])

    # ascisseTF = np.fft.rfftfreq(nPuntiEsatta, T / nPuntiEsatta)
    ascisseTF = np.fft.rfftfreq(nPunti, T / nPunti)

    frequenzeSinistre = []
    frequenzeDestre = []
    for k in range(nMisurePerSpettro):
        trF = np.real(trFourier[k])
        indice = np.argmax(trF)
        frequenzaPicco = ascisseTF[indice]
        altezzaPicco = trF[indice]
        mezzaAltezza = altezzaPicco / 2

        for n in range(indice, -1, -1):
            altezza = trF[n]
            if altezza < mezzaAltezza:
                frequenzeSinistre.append(n)
                break

        for n in range(indice, len(ascisseTF)):
            altezza = trF[n]
            if altezza < mezzaAltezza:
                frequenzeDestre.append(n)
                break

    misureFrequenze = [[] for _ in range(len(frequenzeSinistre))]
    for n in range(len(frequenzeSinistre)):
        trF = np.real(trFourier[n])
        indice = np.argmax(trF)
        fs = 2 * np.pi * ascisseTF[frequenzeSinistre[n]]
        frequenzaPicco = 2 * np.pi * ascisseTF[indice]
        fd = 2 * np.pi * ascisseTF[frequenzeDestre[n]]
        misureFrequenze[n].append(fs)
        misureFrequenze[n].append(frequenzaPicco)
        misureFrequenze[n].append(fd)

    prefissoFrequenze = percorsoFile + "freq_" + str(s)
    moduli.scriviMisureSuFile(misureFrequenze, prefissoFrequenze)

    print(s, misureFrequenze)
    minimi = [
        (misureFrequenze[n][1] - misureFrequenze[n][0])
        for n in range(len(misureFrequenze))
    ]
    picchi = [misureFrequenze[n][1] for n in range(len(misureFrequenze))]
    massimi = [
        (misureFrequenze[n][2] - misureFrequenze[n][1])
        for n in range(len(misureFrequenze))
    ]
    gapAtteso = attesi[s] - attesi[0]

    gapAttesi.append(gapAtteso)
    gapMisurati.append(picchi[0])
    gapErroriMin.append(minimi[0])
    gapErroriMax.append(massimi[0])

    ascisseTF = [(2 * np.pi * f) for f in ascisseTF]

    for k in range(nMisurePerSpettro):
        suffissoFigure = "_" + str(s) + "_" + str(k)
        if misuraUnQubit:
            suffissoFigure += "_sing"
        suffissoFigure += ".png"

        trF = np.real(trFourier[k])
        fs = ascisseTF[frequenzeSinistre[k]]
        hs = trF[frequenzeSinistre[k]]
        fd = ascisseTF[frequenzeDestre[k]]
        hd = trF[frequenzeDestre[k]]

        font = {"family": "serif", "weight": "normal", "size": 44}

        matplotlib.rc("font", **font)

        plt.figure(figsize=(15, 10))
        plt.subplots_adjust(top=0.95, bottom=0.18, left=0.15, right=0.95)
        ax = plt.gca()
        ax.set_xlim((0, 15))
        plt.xlabel("Energy in units of $g^2 / 2$")
        plt.ylabel("$|DFT|$")

        plt.plot(ascisseTF, trF, "bo")
        plt.vlines(
            x=[ascisseTF[np.argmax(trF)], fs, fd],
            ymin=0,
            ymax=[np.max(trF), hs, hd],
            colors=["r", "g", "g"],
        )
        plt.savefig(percorsoFile + "TF" + suffissoFigure)
        plt.close()


font = {"family": "serif", "weight": "normal", "size": 45}

matplotlib.rc("font", **font)

nomeFileGap = percorsoFile + "gap" + str(nPlacchette)
if misuraUnQubit:
    nomeFileGap += "_sing"
nomeFileGap += ".png"
plt.figure(figsize=(15, 10))
plt.xlabel("Energy gaps in units of $g^2 / 2$")
plt.ylabel("Measurements", labelpad=25)
etichetteGap = [str(n) for n in range(1, nStati)]
plt.yticks(ticks=[])
plt.subplots_adjust(top=0.95, bottom=0.18, left=0.15, right=0.95)
plt.errorbar(
    x=gapMisurati,
    y=[n for n in range(1, nStati)],
    xerr=[gapErroriMin, gapErroriMax],
    fmt="o",
    markersize=10,
    linewidth=2,
)
plt.vlines(x=gapAttesi, ymin=0, ymax=nStati, colors=["r"], linewidths=3)
plt.savefig(nomeFileGap)
plt.close()

print(gapAttesi)

rapporti = [gapMisurati[n] / gapAttesi[n] for n in range(len(gapMisurati))]
erroriMinRapporti = [gapErroriMin[n] / gapAttesi[n] for n in range(len(gapMisurati))]
erroriMaxRapporti = [gapErroriMax[n] / gapAttesi[n] for n in range(len(gapMisurati))]
indici = [(n + 1) for n in range(len(gapMisurati))]

nomeFileRapp = percorsoFile + "rapp" + str(nPlacchette)
if misuraUnQubit:
    nomeFileRapp += "_sing"
nomeFileRapp += ".png"
plt.figure(figsize=(15, 10))
plt.xlabel("n")
plt.xticks(indici, [str(ind) for ind in indici])
plt.ylabel("$R_n$")
plt.subplots_adjust(top=0.95, bottom=0.15, left=0.15, right=0.95)
plt.errorbar(
    x=indici,
    y=rapporti,
    yerr=[erroriMinRapporti, erroriMaxRapporti],
    fmt="s",
    markersize=25,
    linewidth=3,
)
plt.hlines(y=1, xmin=0, xmax=len(indici) + 1, colors=["r"], linewidths=4)
plt.savefig(nomeFileRapp)
plt.close()


differenze = [(gapMisurati[n] - gapAttesi[n]) for n in range(len(gapMisurati))]

nomeFileDiff = percorsoFile + "diff" + str(nPlacchette)
if misuraUnQubit:
    nomeFileDiff += "_sing"
nomeFileDiff += ".png"
plt.figure(figsize=(15, 10))
plt.xlabel("n")
plt.xticks(indici, [str(ind) for ind in indici])
plt.ylabel("$R_n$")
plt.subplots_adjust(top=0.95, bottom=0.15, left=0.16, right=0.95)
plt.errorbar(
    x=indici,
    y=differenze,
    yerr=[gapErroriMin, gapErroriMax],
    fmt="s",
    markersize=25,
    linewidth=3,
)
plt.hlines(y=0, xmin=0, xmax=len(indici) + 1, colors=["r"], linewidths=4)
plt.savefig(nomeFileDiff)
plt.close()
