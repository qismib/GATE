import numpy as np
from scipy.optimize import minimize


def salvaBackupStato(nomeFile, stato):
    file = open(nomeFile, "a")
    file.write(str(stato) + "\n")
    file.close()
    print("Stato", stato, "salvato in", nomeFile)


#### Simulazione classica
def statoGenericoReale(entrate):
    if np.linalg.norm(entrate) == 0:
        return entrate

    stato = entrate.copy()
    stato = np.multiply(stato, 1 / np.linalg.norm(stato))
    return stato


def valoreMedio(stato, operatore):
    valore = np.matmul(operatore, stato)
    valore = np.real(np.vdot(stato, valore))
    return valore


def varianza(stato, operatore):
    opQuadro = np.matmul(operatore, operatore)
    var = valoreMedio(stato, opQuadro) - (valoreMedio(stato, operatore)) ** 2
    return var


def ortogonalizza(vettore, vettoriOrtogonali):
    if len(vettore) <= len(vettoriOrtogonali):
        print("ortogonalizza(): non esiste un altro vettore ortogonale")
        return -1

    vettOrtogonale = vettore.copy()
    for i in range(len(vettoriOrtogonali)):
        proiezione = np.vdot(vettore, vettoriOrtogonali[i]) / np.vdot(
            vettoriOrtogonali[i], vettoriOrtogonali[i]
        )
        vettoreProiettato = np.multiply(-proiezione, vettoriOrtogonali[i])
        vettOrtogonale = np.add(vettOrtogonale, vettoreProiettato)

    return vettOrtogonale


_nChiamate_ = 0
_nDuplicati_ = 0


# La varianza è molto veloce ma produce un bias troppo grande
# perchè i dati vengono distribuiti con lunghe code. La stima è
# a 6 dev std dal valore vero. Anche la media non torna
def funzioneCosto(entrate, hamiltoniana, statiInferiori=[], soppressione=20):
    stato = statoGenericoReale(entrate)

    global _nChiamate_
    _nChiamate_ += 1
    costo = varianza(stato, hamiltoniana)  # valoreMedio(stato, hamiltoniana)
    for i in range(len(statiInferiori)):
        sovrapposizione = np.vdot(statiInferiori[i], stato)
        costo += soppressione * abs(sovrapposizione)
    # print("funzioneCosto()", _nChiamate_, costo)
    return costo


def trovaAutostato(hamiltoniana, statiInferiori):
    dim = len(hamiltoniana)
    gdl = dim
    rng = np.random.default_rng()
    copiaStatiInferiori = statiInferiori.copy()

    trovatoNuovoMinimo = False
    tentativi = 1000
    numeroDinamicoTentativi = True
    soppressione = 0  # sembra che aumenti il costo computazionale per il vvqe
    while not trovatoNuovoMinimo:
        for t in range(tentativi):
            parametriIniziali = rng.random((gdl,))
            parametriIniziali = [(p - 0.5) for p in parametriIniziali]
            parametriIniziali = ortogonalizza(parametriIniziali, statiInferiori)
            parametriMinimi = minimize(
                funzioneCosto,
                parametriIniziali,
                (hamiltoniana, copiaStatiInferiori, soppressione),
                method="BFGS",
                options={"xrtol": 0.001},
            )

            minimo = funzioneCosto(parametriMinimi.x, hamiltoniana, copiaStatiInferiori)

            erroreAccettabile = 0.01
            stato = statoGenericoReale(parametriMinimi.x)
            var = varianza(stato, hamiltoniana)

            if numeroDinamicoTentativi and var < erroreAccettabile:
                break

        if not (parametriMinimi.success and var < erroreAccettabile):
            print(
                "trovaAutostato(): minimizzazione fallita (",
                parametriMinimi.message,
                ") Costo = ",
                minimo,
                "; Varianza = ",
                var,
                sep="",
            )
            continue

        trovatoNuovoMinimo = True
        print(
            "trovaAutostato(): minimizzazione terminata in",
            t + 1,
            "passaggi. Varianza =",
            var,
        )
        for i in range(len(statiInferiori)):
            sogliaSovr = 0.1
            sovrapposizione = abs(np.vdot(statiInferiori[i], stato))
            if sovrapposizione > sogliaSovr:
                print(
                    "trovaAutostato(): individuato uno stato troppo vicino agli altri autostati. Si ripete la minimizzazione\n",
                    i,
                    sovrapposizione,
                )
                # soppressione += 10
                trovatoNuovoMinimo = False
                global _nDuplicati_
                _nDuplicati_ += 1
                break

    return stato


def trovaTuttiAutostati(hamiltoniana):
    dim = len(hamiltoniana)
    autostati = []
    for i in range(dim):
        stato = trovaAutostato(hamiltoniana, autostati)
        autostati.append(stato)

    return autostati


def trovaTuttiAutostatiMP(hamiltoniana, autovettoriMP):
    autovettori = trovaTuttiAutostati(hamiltoniana)

    for i in range(len(autovettori)):
        autovettoriMP[i] = autovettori[i]
