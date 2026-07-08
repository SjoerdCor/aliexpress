---
status: accepted
---

# Negatieve tevredenheid voor geschonden vermij-voorkeuren

Een leerling met uitsluitend vermij-voorkeuren ("liever niet met") kreeg een
gewicht-afhankelijke tevredenheid (`1.0 + F(honored_sum)`): bij gewicht 1 precies 0%, bij
gewicht 2 −200%, enzovoort. Dat onderschatte zijn ontevredenheid én stuurde de lexmaxmin
verkeerd — een geschonden vermij-voorkeur telde als even erg als een misgelopen
graag-met-voorkeur (0%), terwijl geschonden worden een actieve last is.

## Beslissing: asymmetrische schaal met een sprong

De tevredenheidsschaal wordt asymmetrisch:

- Positieve voorkeuren ("graag met") lopen van 0% (geen enkele ingewilligd) tot 100% (alle
  ingewilligd) — ongewijzigd.
- Een leerling zonder positieve voorkeuren is 100% zolang hij van iedereen wordt
  weggehouden, maar *springt* bij de eerste geschonden vermij-voorkeur onder 0%, in een
  per-leerling genormaliseerde, verzadigende band tot −100% (alle vermij-voorkeuren
  geschonden). "Veilig weggehouden" is daarmee de enige volledig-tevreden uitkomst.

De sprong (van +100% direct naar negatief bij de eerste schending) is bewust: een vermij-wens
gaat over vermijden, dus élke schending maakt de leerling ontevreden. De band is per leerling
genormaliseerd op zijn eigen slechtste geval (`raw / |F(worst_sum)|`), zodat één geschonden
vermij-wens −100% geeft ongeacht het gewicht; het gewicht verdeelt de band alleen tussen
meerdere vermij-wensen, met dezelfde verzadiging als de positieve kant.

De maat leeft in één gedeelde helper `_normalize_and_bound` in `satisfaction.py`, aangeroepen
door zowel de rapportage (`engine._float_satisfaction`) als de integer-scoretabel van het
model (`modelbuilder._scaled_satisfaction`). Zo lopen het gerapporteerde percentage en de
waarde die het model optimaliseert per definitie niet uiteen — de duplicatie die deze twee
plekken eerder liet divergeren is daarmee weg (de docstrings verwezen al naar deze
niet-bestaande helper).

## Overwogen alternatieven

- **Continue schaal [−100%, +100%] zonder sprong** (`1 − 2·raw/F(worst_sum)`) — afgewezen.
  Die laat een leerling die van 4 van zijn 5 gevreesde klasgenoten is weggehouden nog hoog
  scoren. Gewenst is juist dat élke schending de leerling direct in negatief gebied brengt.
- **Positieve en negatieve as samenvoegen tot één herschaling (`P − 2·N`), ook voor gemengde
  leerlingen** — afgewezen: te ingrijpend. De enige echte special-case is "geen positieve
  voorkeuren"; de gewone tak (inclusief gemengde leerlingen) blijft ongewijzigd.

## Consequenties

- Tevredenheid kan negatief zijn. De lexmaxmin ondersteunt dit al: `lower_bound` in
  `strategies.py` is data-gedreven uit de per-leerling bounds, met een comment die expliciet
  "a student with only violated negative wishes" noemt. De plateau-logica tilt de laagste
  eerst op, dus een op −100% geplaatste vermij-leerling wordt als eerste opgetild.
- Integratietests die vermij-only-scenario's vastpinnen moeten opnieuw gepind worden; sommige
  optima verschuiven (een geschonden vermij-leerling wordt nu vóór een graag-met-0%-leerling
  opgetild). Dit is het beoogde gedrag, geen regressie.
