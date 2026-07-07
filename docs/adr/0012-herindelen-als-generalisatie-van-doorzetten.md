---
status: accepted
---

# Herindelen als generalisatie van doorzetten (één engine, jaarlaag als partitie)

De app doet vandaag **doorzetten**: één instromende jaarlaag wordt verdeeld over de
gecombineerde groepen van de volgende bouw, terwijl de blijvende oudere kinderen de vaste,
niet-beïnvloedbare bezetting van die bestemmingsgroepen vormen en de oudste jaarlaag de school
verlaat. Er kwam een verzoek voor **herindelen**: een aantal bestaande groepen — welke bouw dan ook
(bovenbouw 6-7-8, maar net zo goed kleutergroepen of twee groep-5'en) — waaruit kinderen zijn
vertrokken opnieuw indelen over diezelfde groepen. De zorg: de app houdt nu geen rekening met
spreiding over jaarlagen, dus een uitkomst met "de ene groep vooral jaarlaag 6, de andere vooral 7"
zou onacceptabel zijn.

Naïef lijkt dit een tweede systeem. Maar de solver kent al een onderscheid tussen *de nieuwe
jaarlaag* (`year`) en *totaal*, én een anti-clique op de herkomstgroep. Voor de solver is doorzetten
in feite het geval met precies één bewegende jaarlaag; de uitstroom en de vaste achterblijvers zitten
volledig in de dataprep (bezetting via `groups_to`).

## Beslissing: één gegeneraliseerde engine, jaarlaag als partitie van de te verdelen leerlingen

- **Eén engine, geen tweede solver-pad.** De solver splitst de per-cohort-balans op de *distinct*
  jaarlagen onder de bewegende leerlingen (`Jaarlaag`, altijd uit EDEXML). Bij doorzetten is er één
  bewegende jaarlaag (of, bij data zonder jaarlaag, één impliciet cohort) → de constraints reduceren
  wiskundig tot het huidige gedrag. Bij herindelen zijn er één of meer jaarlagen. Geen mode-vlag in
  de solver: het aantal distinct jaarlagen stuurt het.
- **Per-jaarlaag balans met één gedeelde familie-slack, gezamenlijk opgelost.** De twee
  `year`-constraints ([`_constraint_equal_new_students`](../../src/aliexpress/solver/problemsolver.py),
  [`_constraint_equal_boys_girls`](../../src/aliexpress/solver/problemsolver.py)) worden per
  jaarlaag geïnstantieerd — zowel **aantal** als **jongen/meisje** binnen elke jaarlaag. Per familie
  is er **één gedeelde slack** (vaste naam, zoals nu): de familie relaxeert naar de zwaarste jaarlaag,
  zodat de slack niet per jaarlaag verschilt en de slack-namen stabiel blijven. De `_total`-
  constraints blijven ongewijzigd en bewaken de totale klassenbalans. Alles zit in hetzelfde LP en
  wordt in één solve gezamenlijk opgelost; de per-jaarlaag-eisen komen bovenop de totaal-eisen, niet
  in de plaats daarvan.
- **Iedereen die meedoet wordt herverdeeld; bezetting 0.** In herindelen is er geen vaste
  bezetting: het "Wie gaat mee"-scherm houdt zijn bestaande betekenis (uitgevinkt = valt weg — een
  verlenger óf een vertrekker). Een enkele leerling toch vastzetten kan via de bestaande
  "Niet in"-uitsluiting op alle andere groepen; daar is geen apart mechanisme voor.
- **Anti-clique blijft actief** in beide modi (herindelen mag dus bewust remixen).
- **Bron van de jaarlaag: EDEXML** (`jaargroep`, nu al ingelezen maar ongebruikt). Het Excel-pad
  krijgt voorlopig geen jaarlaag-kolom.
- **Verdeelmodus wordt bij het aanmaken van het proces gekozen** en opgeslagen als bestand in de
  procesmap (zoals `input_method.json`; geen DB-migratie). De modus stuurt de wizard-branch.
- **Bestemmingsgroepen en kandidaten worden bij de EDEXML-upload afgeleid.** In herindelen kies je
  daar de te herindelen **groepen** (i.p.v. één instroom-jaargroep); kandidaten = alle leerlingen
  in die groepen, `groups_to` = diezelfde groepen met bezetting 0. Het bestaande groepen-scherm
  bevestigt/verfijnt die afgeleide groepen, net als nu.

### Waarom

De satisfactie, voorkeuren, het LP-substraat, de adaptieve balans, opslag, login, sociogram en
rapportage-substraat blijven ongewijzigd. Eén engine voorkomt codeduplicatie en het uit elkaar
groeien van twee paden. Doorzetten wordt het gedegenereerde geval, dus bestaand gedrag blijft
wiskundig identiek.

## Overwogen alternatieven

- **Twee gescheiden solver-paden.** Raakt de bestaande solver niet, maar geeft duplicatie, dubbel
  onderhoud en divergentie-risico. Afgewezen.
- **Achterblijvers als vaste bezetting per jaarlaag** (hergebruik van het "Wie gaat mee"-scherm om
  individuen vast te zetten). Verworpen: het punt van herindelen is juist dat sommige leerlingen
  níét meedoen (vertrekkers). Individueel vastzetten is zeldzaam en de "Niet in"-truc volstaat.
- **Nieuw optimalisatiedoel "minimaliseer verhuizingen"** (minimale verstoring). Afgewezen: de
  gekozen filosofie is balans + voorkeuren maximaliseren, waarbij verplaatsen een gevolg is, geen
  doel. Scheelt een nieuw, moeilijk af te wegen doel.
- **Instelbare/losse anti-clique in herindelen.** Geparkeerd als kleine bijstelling voor later
  (zie Consequenties).

## Consequenties

- **`feasibility.py` blijft ongewijzigd.** Doordat de per-jaarlaag constraints één *gedeelde*
  familie-slack delen (vaste naam, zoals nu), blijven `RELAXATION_WEIGHTS`/`weighted_relaxation` op
  stabiele namen matchen — geen dynamische slack-namen, geen prefix-matching, geen normalisatie nodig.
  Dit was aanvankelijk voorzien als het meest precieze stuk werk; het vervalt door de gedeelde slack.
  Alleen de interne hulp-variabelen (per-jaarlaag telling/min/max) krijgen een jaarlaag-suffix voor
  unieke pulp-namen, maar die worden nergens op naam gematcht.
- **Doorzetten moet aantoonbaar ongewijzigd blijven.** De integratietests toetsen exacte
  tevredenheid; ze moeten groen blijven — empirisch bewijzen, niet aannemen.
- **Watchpunt:** de adaptieve balans start streng en trekt de huidige groepen richting een
  gelijkmatige spreiding (de anti-clique versoepelt slechts tot haalbaarheid). Als dat te ingrijpend
  blijkt, is een losse/instelbare anti-clique de kleine bijstelling.
- **Rapport krijgt een jaarlaag×groep-overzicht** zodat de leerkracht de bereikte spreiding kan
  zien.
- **Prestaties** op realistische omvang (±3 groepen × ~75 kinderen) worden gebenchmarkt.
- **Buiten scope:** jaarlaag-kolom in `voorkeuren.xlsx`, groepen opheffen/samenvoegen, pin-UI.

## Uitbreiding: Herindelen met doorzetten (2026-07-05)

In de praktijk valt een herindeling vaak samen met de jaarovergang: niet de huidige bewoners
van 6-7-8 worden herverdeeld, maar de leerlingen van jaarlagen 5-6-7 komen in de (leeg
startende) huidige groepen van 6-7-8. De bron-populatie valt dan níét samen met de
bestemmingsgroepen — de jaarlaag-5'ers zitten nu in de middenbouw. De bestaande
herindeel-invoer (kies fysieke groepen, populatie = hun bewoners) kan dit niet uitdrukken.

### Beslissing: subvariant binnen Herindelen, populatie op jaarlaag

- **Geen derde verdeelmodus.** De backend is aantoonbaar identiek aan herindelen (de
  anti-clique leidt herkomstgroepen uit de leerlingdata zelf af; de per-jaarlaag balans
  werkt op de distinct jaarlagen onder de bewegende leerlingen). Een derde modus zou
  alleen de wizard dupliceren. "Met doorzetten" wordt bínnen de herindeel-tak gekozen,
  direct na de EDEXML-upload; `Verdeelmodus` blijft tweewaardig.
- **Flow spiegelt doorzetten: jaarlagen eerst, dan groepen.** De leerkracht kiest de
  bewegende jaargroepen (bijv. 5-6-7) en daarna handmatig de bestemmingsgroepen.
  Aaneengesloten jaarlagen zijn een zachte conventie, niet hard afgedwongen. Bewust géén
  slimme voorinvulling van jaarlagen uit de gekozen groepen: door de volgorde
  (jaarlagen eerst) valt er niets af te leiden, en groepen aanvinken is een kleine moeite.
- **Populatie = alle leerlingen in de gekozen jaarlagen, school-breed.** Geen
  jaarlaag×bron-groep-scoping: het gangbare geval heeft één stroom, en over-selectie
  (parallelle stroom die elders heen moet) wordt opgevangen door de bestaande
  "Wie gaat mee"-stap (vertrekkers uitvinken) — het vangnet bestaat al.
- **Alleen via EDEXML**, net als gewoon herindelen: de jaarlaag komt alleen daaruit.

### Verworpen alternatieven

- **Derde verdeelmodus** ("Herindelen met doorzetten" als eigen knop bij procesaanmaak):
  dupliceert de volledige herindeel-tak voor één afwijkende invoerstap.
- **Volledige unificatie** (herindelen = altijd populatie + bestemming los kiezen): maakt
  het gangbare geval (zelfde leerlingen terug) een stap omslachtiger.
- **Jaarlaag×bron-groep-selectie**: dekt parallelle stromen, maar extra klikwerk en
  foutkans voor het gangbare geval; de roster-stap dekt het randgeval al.

### Consequenties

- **Solver, feasibility en rapportage: nul wijzigingen** (geverifieerd: `_cliques` leest
  `Stamgroep` uit de leerlingdata, niet uit een lijst; de herkomst×bestemming-crosstab
  blijft betekenisvol als herkomst- en bestemmingslabels verschillen).
- De enige inhoudelijke delta zit in de dataprep (`candidatedetermination`): kandidaten
  op jaarlaag selecteren, en `groups_from` = de échte herkomstgroepen van die jaarlagen
  (voedt de herkomst-dropdown van de roster-stap), niet de bestemmingsgroepen.
