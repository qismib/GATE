import moduliEvoluzionePlacchette as moduli
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy.optimize import curve_fit

DATA_PATH = "../../../data"


def sovrapposizione(stati, coefficienti):
    v = [0] * len(stati)
    for idx, state in enumerate(stati):
        v = np.add(v, np.multiply(state, coefficienti[idx]))

    return v


def clusterJackknife(misure):
    cluster = []
    media = np.mean(misure)
    N = len(misure)

    for k in range(N):
        binJack = media - ((misure[k] - media) / (N - 1))
        cluster.append(binJack)

    return cluster


def gaussiana(x, media, sigma, altezza):
    return altezza * np.e ** (-((x - media) ** 2) / (2 * sigma**2))


parametriGaussiane = [
    [5.94447581, 0.12128139, 26.56598046],
    [5.94409544, 0.12025162, 26.70177117],
    [5.94360316, 0.12339673, 26.14287223],
    [5.9409017, 0.12431001, 26.23920353],
    [5.94372287, 0.11974104, 26.52011432],
]

leggiMisureDaFile = True
x = 1.5
H = [
    [0, -2 * x, -2 * x, 0],
    [-2 * x, 3, 0, -x / 2],
    [-2 * x, 0, 3, -x / 2],
    [0, -x / 2, -x / 2, 3],
]
autovalori, matriceAutovettori = np.linalg.eigh(H)
matriceAutovettoriInversa = np.linalg.inv(matriceAutovettori)
nStati = len(H)

stati = [
    [0.808992047467855, 0.4101442096668511, 0.4101115648454173, 0.09550967904126201],
    [
        -0.20200917702552543,
        0.30548582993873763,
        -0.12773951931771707,
        0.9217121651082778,
    ],
    [
        -0.06861852324779168,
        -0.6452995532134647,
        0.7024301247942335,
        0.29235578439865517,
    ],
    [0.5480385666960621, -0.5706671071253255, -0.5663434088132931, 0.23075512031735548],
]

rng = np.random.default_rng()
tentativi = 20
shots = 100
picchiMisurati = []

for s in range(1, 2):
    picchiMisurati.append([])

    coefficienti = [0] * nStati
    coefficienti[0] = 0.5**0.5
    coefficienti[s] = 0.5**0.5
    vIniziale = sovrapposizione(stati, coefficienti)
    vIniziale = np.multiply(vIniziale, 1 / np.linalg.norm(vIniziale))

    durateEvT = [40]  # [20, 40, 100, 250, 500, 1000]
    freqMin = []
    freqPicco = []
    freqMax = []
    probAggregata = []

    for T in durateEvT:
        risoluzione = 2 * np.pi / T
        print("Risoluzione:", risoluzione)

        dt = 0.1
        nPuntiEsatta = int(T / dt)
        ascisseEvEsatta = [(T * n / nPuntiEsatta) for n in range(nPuntiEsatta)]

        nMis = 2  # si misura solo primo qubit
        statoEv = moduli.evoluzioneTemporaleStato(vIniziale, H, ascisseEvEsatta)
        ordinateProb = [[] for n in range(nMis)]
        for n in range(len(statoEv)):
            somma0 = (abs(statoEv[n][0]) ** 2) + (abs(statoEv[n][1]) ** 2)
            somma1 = (abs(statoEv[n][2]) ** 2) + (abs(statoEv[n][3]) ** 2)
            ordinateProb[0].append(somma0)
            ordinateProb[1].append(somma1)

        nPuntiPrecisa = 10000
        ascisseEvPrecisa = [(T * n / nPuntiPrecisa) for n in range(nPuntiPrecisa)]
        statoEvPreciso = moduli.evoluzioneTemporaleStato(vIniziale, H, ascisseEvPrecisa)
        ordinateProbPrecise = [[] for _ in range(nMis)]
        for n in range(len(statoEvPreciso)):
            somma0 = (abs(statoEvPreciso[n][0]) ** 2) + (abs(statoEvPreciso[n][1]) ** 2)
            somma1 = (abs(statoEvPreciso[n][2]) ** 2) + (abs(statoEvPreciso[n][3]) ** 2)
            ordinateProbPrecise[0].append(somma0)
            ordinateProbPrecise[1].append(somma1)

        percorsoFile = f"{DATA_PATH}/errori/"
        if not leggiMisureDaFile:
            #### Simulazione esatta

            for j in range(tentativi):
                ordinateFlutt = [[] for _ in range(nMis)]

                for n in range(len(statoEv)):
                    probEntrate = [(abs(c) ** 2) for c in statoEv[n]]
                    estrazioni = rng.multinomial(n=shots, pvals=probEntrate)
                    estrazioni = [(e / shots) for e in estrazioni]

                    somma0 = estrazioni[0] + estrazioni[1]
                    somma1 = estrazioni[2] + estrazioni[3]

                    ordinateFlutt[0].append(somma0)
                    ordinateFlutt[1].append(somma1)

                moduli.scriviMisureSuFile(
                    ordinateFlutt, percorsoFile + "EvTemp" + str(j)
                )
                probAggregata.append(ordinateFlutt[0])

        else:
            for j in range(tentativi):
                ordinateFlutt = moduli.leggiMisureDaFile(
                    percorsoFile + "EvTemp" + str(j), nMis, nPuntiEsatta
                )
                probAggregata.append(ordinateFlutt[0])

        probAggregata = np.transpose(probAggregata)
        medieProb = [np.mean(c) for c in probAggregata]
        erroriProb = [(np.std(c) / (len(c) ** 0.5)) for c in probAggregata]

        font = {"family": "serif", "weight": "normal", "size": 40}

        matplotlib.rc("font", **font)

        prefissoFigure = percorsoFile + "fluttuazioniEvT"
        suffissoFigure = (
            "_" + str(T) + "_" + str(s) + "_" + str(shots) + "_shots" + ".png"
        )

        plt.figure(figsize=(15, 10))
        plt.subplots_adjust(top=0.98, bottom=0.15, left=0.11, right=0.95)
        plt.xlim(0, 10)
        plt.xlabel("t [$2 / g^2$]")
        # plt.ylabel("Probability")
        plt.plot(ascisseEvPrecisa, ordinateProbPrecise[0], "k")
        plt.errorbar(
            x=ascisseEvEsatta,
            y=medieProb,
            yerr=erroriProb,
            marker="o",
            color="b",
            linestyle="",
        )
        plt.savefig(prefissoFigure + "ProbGrande" + suffissoFigure)
        plt.close()

        #### Trasformate di Fourier
        ascisseTF = np.fft.rfftfreq(nPuntiEsatta, T / nPuntiEsatta)

        trFourier = np.fft.rfft(medieProb)
        trFourier[0] = 0
        for m in range(len(trFourier)):
            trFourier[m] = abs(trFourier[m])
        trFourier = np.real(trFourier)

        probCluster = []
        for k in range(len(probAggregata)):
            probCluster.append(clusterJackknife(probAggregata[k]))
        probCluster = np.transpose(probCluster)

        trFourierCluster = []
        for k in range(len(probCluster)):
            trC = np.fft.rfft(probCluster[k])
            trC[0] = 0
            for m in range(len(trC)):
                trC[m] = abs(trC[m])
            trC = np.real(trC)

            trFourierCluster.append(trC)

        trFourierCluster = np.transpose(trFourierCluster)
        erroriTrFourier = []
        for k in range(len(trFourierCluster)):
            N = len(trFourierCluster[k])
            varianzaJack = 0

            for m in range(N):
                varianzaJack += (trFourierCluster[k][m] - trFourier[k]) ** 2

            varianzaJack *= (N - 1) / N
            erroriTrFourier.append(varianzaJack ** (0.5))

        # print(erroriTrFourier)

        ascisseTF = [(2 * np.pi * c) for c in ascisseTF]
        nMin = 30
        nMax = 50
        xDaFittare = ascisseTF[nMin:nMax]
        yDaFittare = trFourier[nMin:nMax]
        erroriDaFittare = erroriTrFourier[nMin:nMax]
        # parametri = [0.9, 0.1, 25]
        parametri, covarianza = curve_fit(
            gaussiana, xDaFittare, yDaFittare, p0=[6, 1, 10], sigma=erroriDaFittare
        )

        nXPrecise = 500
        xPrecise = [
            min(xDaFittare) + ((max(xDaFittare) - min(xDaFittare)) * x / nXPrecise)
            for x in range(nXPrecise)
        ]
        valoriGaussiana = [
            gaussiana(x, parametri[0], parametri[1], parametri[2]) for x in xPrecise
        ]

        plt.figure(figsize=(15, 10))
        plt.subplots_adjust(top=0.98, bottom=0.15, left=0.11, right=0.95)
        plt.xlim(0, 10)
        plt.xlabel("$E_k$ [$g^2 / 2$]")
        plt.ylabel("$|A_k|$", labelpad=20)
        plt.errorbar(
            x=ascisseTF,
            y=trFourier,
            yerr=erroriTrFourier,
            marker=".",
            color="b",
            linestyle="--",
        )
        plt.plot(xPrecise, valoriGaussiana, "r")
        # plt.vlines(x = [ascisseTF[np.argmax(trFourier)]], ymin = 0, ymax = [np.max(trFourier)], colors = ["r", "k", "k"])
        plt.savefig(prefissoFigure + "TF" + suffissoFigure)
        plt.close()

        print(parametri)

        indiceFreqMax = np.argmax(trFourier)
        plt.figure(figsize=(15, 10))
        plt.subplots_adjust(top=0.98, bottom=0.15, left=0.11, right=0.95)
        plt.xlim(5.5, 6.5)
        plt.xlabel("Energy in units of $g^2 / 2$")
        plt.ylabel("$|DFT|$", labelpad=20)

        colori = ["k", "c", "m", "tab:orange", "tab:pink"]
        for k in range(len(parametriGaussiane)):
            p = parametriGaussiane[k]
            valoriGaussiana = [gaussiana(x, p[0], p[1], p[2]) for x in xPrecise]
            plt.plot(xPrecise, valoriGaussiana, colori[k])

        plt.vlines(
            x=[
                ascisseTF[indiceFreqMax],
                ascisseTF[indiceFreqMax - 1],
                ascisseTF[indiceFreqMax + 1],
                5.891,
            ],
            ymin=0,
            ymax=[np.max(trFourier)],
            colors=["r", "g", "g", "tab:olive"],
        )
        plt.savefig(prefissoFigure + "TFtutte" + suffissoFigure)
        plt.close()
