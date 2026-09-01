# Metingen: tijdschatting (ETA) voor de processing-pagina

**Status:** herkalibratie na `perf: speed up proven balance optimization`, 2026-08-31.
Deze meting vervangt de constanten uit slice 7. De drie eerder gemeten modi zijn de
eerste dataset; daarna zijn korte herhalingen en twee schaalvarianten toegevoegd. De
absolute tijden blijven machine- en run-afhankelijk, maar de gate en de richting van de
schatting zijn op de volledige dataset gecontroleerd.

## Meetopstelling

- Pijplijn: `main.distribute_students_from_data` →
  `engine.solve_within_minimal_relaxation`, dus hetzelfde automatische pad als in de
  webapp. `solve_with_fixed_balance` is niet gemeten.
- Stage-tijden komen uit `ProgressListener.stage_finished`: `floor`, `balance` en
  `satisfaction`. De per-leveltijden komen uit `PlateauOutcome.seconds`.
- Eén Windows 11-machine, CP-SAT met `num_workers=8`.
- De herhalingen gebruiken dezelfde invoer waar dat vermeld staat. De twee nieuwe
  schaalvarianten komen uit `main_redistribute_and_forward` en gebruiken de standaard
  balansgrenzen voor die tijdelijke meetinvoer; er zijn geen productieresultaten
  overschreven.

## Dataset 1: reeds gemeten drie modi

Deze waarden komen uit de post-performance metingen die in ADR-0018 zijn vastgelegd.
Ze zijn als uitgangspunt behouden, niet opnieuw geïnterpreteerd als nieuwe runs.

| modus en invoer | runs | vloer | balans | tevredenheid | totaal | sat/balans |
|---|---:|---:|---:|---:|---:|---:|
| Doorzetten, 35 leerlingen | 3 | 0,12–0,20 s | 0,90–0,93 s | 2,37–2,75 s | 3,45–3,93 s | 2,6–3,1 |
| Herindelen, 35 leerlingen | 3 | 0,05–0,07 s | 0,31–0,35 s | 0,67 s | 1,07–1,10 s | 1,9–2,2 |
| Herindelen met doorzetten, 72 leerlingen, harde balansgrenzen | 2 | 3,5–16,7 s | 554,6–751,9 s | 480,0–569,4 s | 1.038,1–1.338,1 s | 0,6–1,0 |

De derde modus heeft hier een andere regime: de balansfase zelf is de dominante, sterk
variabele bewijsstaart. Daarom mag de fase-B-schatting niet de volledige balansduur nog
eens zesmaal vermenigvuldigen.

## Dataset 2: herhalingen en hardheidsvarianten

De eerste twee regels zijn actuele herhalingen op dezelfde opgeslagen invoer. De laatste
twee zijn dezelfde realistische generator met meer leerlingen; ze laten zien waar de
45-seconden-gate omslaat.

| modus en variant | vloer | balans | tevredenheid | rondes | totaal | sat/balans |
|---|---:|---:|---:|---:|---:|---:|
| Doorzetten, 35 leerlingen, herhaling | 0,21 s | 1,05 s | 2,59 s | 11 | 3,88 s | 2,5 |
| Herindelen, 35 leerlingen, herhaling | 0,11 s | 0,34 s | 0,78 s | 6 | 1,27 s | 2,3 |
| Herindelen met doorzetten, 48 leerlingen | 1,25 s | 13,28 s | 46,90 s | 7 | 61,51 s | 3,5 |
| Herindelen met doorzetten, 60 leerlingen | 4,68 s | 50,29 s | 265,84 s | 7 | 320,87 s | 5,3 |

Ook in deze twee varianten is de variantie zichtbaar: de eerste zeven rondes van de
60-leerlingenrun duurden achtereenvolgens 14,92 · 56,88 · 51,84 · 48,86 · 26,20 ·
26,92 · 17,05 s. De 48-leerlingenrun had 4,61 · 4,63 · 6,05 · 4,41 · 4,41 · 8,54 ·
6,67 s. Eén gemiddelde of één eerste ronde is dus geen betrouwbare vaste snelheid.

## Kalibratie

### Fase A

Zolang de balansfase nog niet klaar is, blijft de generieke tekst staan:

> Aan het rekenen… dit duurt meestal minder dan een minuut, soms enkele minuten.

### Fase B

Na afronding van de balansfase en vóór de eerste tevredenheidsronde gebruikt de writer:

```text
normale_balans = min(balance_duur, 60)
lange_balans   = max(balance_duur - 60, 0)
resterend      = 6 × normale_balans + 1 × lange_balans
```

De factor 6 is de bovengrens van de gewone sat/balans-verhoudingen (5,3 in de
60-leerlingenvariant), met een kleine marge naar boven. De grens van 60 seconden voorkomt
dat de balansgedomineerde stressruns als 55–75 minuten resterend worden gepresenteerd:
bij de twee harde 72-leerlingenruns komt fase B uit op ongeveer 855–1.052 seconden
(14–18 minuten), tegenover 480–569 seconden werkelijke tevredenheidstijd. Dat is bewust
ruim, maar nog in dezelfde orde van grootte.

### Fase C

Na elke afgeronde tevredenheidsronde blijft de adaptieve schatting gebaseerd op de
langste ronde tot nu toe:

```text
resterende_rondes = max(11 - rondes_klaar, 1)
resterend         = resterende_rondes × langste_rondeduur
```

Elf is het hoogste aantal rondes in de actuele dataset (Doorzetten); de andere modi
zaten op zes of zeven. De langste ronde blijft de veilige keuze omdat CP-SAT-rondes
binnen één run kunnen oplopen. De schatting blijft een ruwe ETA, geen bovengrens of SLA.

De tekst wordt onder 60 seconden naar boven afgerond op tientallen seconden en vanaf
60 seconden naar boven op hele minuten.

## Controle van de 45-seconden-gate

De processing-pagina onthult rijke voortgang wanneer de geschatte resterende tijd `> 45 s`
is of wanneer de verstreken tijd `>= 45 s` is. De backend-formule en de gate gebruiken
dezelfde ruwe seconden; de grens blijft in `templates/processing.html` op 45 seconden.

| datasetgeval | fase-B-ETA met nieuwe formule | werkelijke totale tijd | classificatie |
|---|---:|---:|---|
| Doorzetten, referentie/herhaling | 5,4–6,3 s | 3,45–3,93 / 3,88 s | kort; blijft dicht |
| Herindelen, referentie/herhaling | 1,9–2,1 s | 1,07–1,27 s | kort; blijft dicht |
| Herindelen met doorzetten, 48 | 79,7 s | 61,5 s | lang; estimate onthult vroeg |
| Herindelen met doorzetten, 60 | 301,7 s | 320,9 s | lang; estimate onthult vroeg |
| Herindelen met doorzetten, harde 72 | 855–1.052 s | 1.038–1.338 s | lang; estimate onthult vroeg |

Er is in deze dataset geen vals positieve estimate-gate: de korte modi blijven ruim onder
45 seconden en alle middel/lange varianten zitten erboven. Onafhankelijk daarvan vangt
de verstreken-tijdfallback ieder geval af dat uiteindelijk langer dan 45 seconden duurt,
ook als CP-SAT de eerste estimate onderschat.

## Conclusie

De oude fase-B-formule `12 × balance_duur` was gebaseerd op metingen vóór de performance-
verbetering en is niet meer passend. De nieuwe stukgewijze formule (`6×` tot 60 seconden,
daarna `1×`) en het nieuwe rondesbudget (`11`) werken voor de drie modi, blijven voor
gewone runs rustig en geven bij de gemeten lange runs vroeg rijke voortgang vrij. De grote
run-tot-run-variantie blijft een eigenschap van CP-SAT; daarom blijft de UI-tekst expliciet
een ruwe schatting.

De pre-performance slice-7-metingen en de oorspronkelijke factor 12 zijn als historische
achtergrond terug te vinden in de commitgeschiedenis en ADR-0018; ze zijn niet langer de
kalibratiedataset.
