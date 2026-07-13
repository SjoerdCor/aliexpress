---
status: accepted
---

# Logging-strategie: app-breed, naamloos event-log

De app kent vandaag twee log-kanalen die niets van elkaar weten:

- **Technisch** — `logger.*` op de `aliexpress`-pakketlogger, naar de console en (in productie)
  het bestand `instance/logs/aliexpress.log`. Format: alleen `tijd - niveau - bericht`
  ([`logging_config.py`](../../src/aliexpress/logging_config.py)).
- **Gebruikersgericht** — `on_update`-berichten die als `LogLine`-rijen in de database belanden
  en op de verwerk-pagina worden getoond (Nederlands, per Proces).

Een volledige run langs alle wizard-routes legde drie problemen empirisch bloot:

1. **Geen correlatie.** Het format draagt geen logger-naam, geen thread, geen Proces/run.
   De solve- en sociogram-thread draaien gelijktijdig en hun regels lopen door elkaar; bij
   meerdere scholen tegelijk is een run niet meer uit elkaar te trekken.
2. **Vroege stappen zijn stil bij succes.** `upload_edexml`, `roster` en `not_together` loggen
   niets op het happy path. Bij een melding als "de verkeerde kinderen zaten erin" geeft de log
   niets.
3. **Leerlingnamen op DEBUG in een persistent bestand.** `_load_student_names` logt volledige
   namen op DEBUG, en de productie-file-handler staat óók op DEBUG. Namen van minderjarigen
   belanden zo in één app-breed, niet-roterend logbestand dat buiten de school-isolatie én buiten
   de purge-naad valt.

De eis die de hele afweging bindt: **een diagnostisch spoor moet een procesverwijdering door de
leerkracht overleven** (die verwijdert het Proces soms uit frustratie of om opnieuw te proberen).
Wat een verwijdering overleeft, moet per definitie *buiten* de procesmap leven — dus app-breed.

## Beslissing: één app-breed, naamloos event-log + perishable per-leerling detail

- Er is één **app-breed event-log** dat de inputs, beslissingen en fouten per fase vastlegt,
  met **correlatie** (school / Proces / run / thread / fase) en **nooit leerlingnamen**. Het
  bestand roteert en kent een bewaartermijn.
- **Per-leerling detail** (namen, tevredenheid per kind, vervulde voorkeuren) blijft *perishable*
  in de procesmap (`results.xlsx`, `result_tables.json`) en in de `LogLine`-rijen, en wordt mét
  het Proces opgeruimd. Dat dit verdwijnt bij verwijdering is correct gedrag, geen verlies.
- **Dekking**: per fase één à twee event-regels (aantallen/keuzes) op INFO; in de solve de
  uiteindelijke relaxatie/slack en een tevredenheids-samenvatting op INFO; verbose solver-detail
  (lexmaxmin-iteraties e.d.) op DEBUG.
- **Mechanisme** (implementatie): `%(name)s` + `%(threadName)s` in het format; school/Proces/run/
  fase via een `contextvar` + een `logging.Filter`, zodat geen enkele bestaande `logger.*`-aanroep
  hoeft te veranderen. Achtergrond-threads zetten de context uit `_ThreadContext`, request-context
  uit de sessie.

### Waarom naamloos — de kern

- Het enige spoor dat een verwijdering overleeft, is óók het enige spoor dat álle scholen
  aggregeert en de purge overleeft. Dat is exact de plek waar geen PII van minderjarigen hoort.
  **De survivability-eis maakt naamloosheid noodzakelijk, niet optioneel.**
- **Privacy door constructie boven privacy door correcte threading.** Een naamloos log kan niet
  lekken door een concurrency-bug; per-Proces-isolatie via gedeelde-logger-routing kan dat wel.
  Voor gegevens van minderjarigen kiezen we de garantie die niet kan terugregresseren.
- De naamloze inhoud (school/Proces/run + beslissing/fout) is juist het deel dat ná verwijdering
  nog nuttig is — het beantwoordt "waar" en "waarom". Het deel dat verdwijnt (wélke leerling) is
  precies wat AVG niet in een overlevend bestand wil, en wat je voor een *structurele* bug zelden
  nodig hebt.

## Overwogen alternatieven

- **Per-Proces logbestand in de procesmap + purge.** Lost tenant-scheiding en opruimen op, en
  geeft correlatie gratis (één bestand = één run). Maar het verdwijnt mét het Proces — het faalt
  juist in het verwijder-scenario dat de eis stelt — en het leunt voor privacy op foutloze
  per-thread handler-routing van de gedeelde logger.
- **Namen alleen naar de dev-console, prod-file op INFO.** Werkt, maar de privacywaarborg hangt
  aan blijvende discipline dat geen enkele naam ooit op INFO of hoger gelogd wordt.
- **Namen in het overlevende log, bestand beveiligen.** Maximale naspeurbaarheid, maar het grootste
  PII-risico op de meest blootgestelde, langst-levende plek.
- **Ondoorzichtige per-leerling-id i.p.v. naam.** De `matching_key` is een genormaliseerde naam
  (nog herleidbaar); een per-run-index voegt complexiteit toe voor weinig, omdat het per-leerling
  detail al in de artefacten staat.

## Consequenties

- `logging_config.py` krijgt het uitgebreide format (naam + thread), de `contextvar`/`Filter`, en
  een `TimedRotatingFileHandler` (dagelijks, `backupCount=90` → 90 dagen bewaartermijn) op **INFO**.
  DEBUG blijft alleen op de console tijdens ontwikkeling.
- De vroege wizard-stappen krijgen naamloze event-regels. De naam-dumpende DEBUG-regels
  (`_load_student_names`, `processor.student_display`) worden **verwijderd** — geen diagnostische
  waarde én de PII-lek; `input_writer`'s inhoudsloze `"Data ingevuld"` gaat mee weg.
- `main.py` logt de uiteindelijke relaxatie/slack en een tevredenheids-samenvatting bij afronden.
- **De kwetsbare solve wordt zichtbaar in het overlevende (INFO) log.** Per lexmaxmin-niveau worden
  de *headline-metrics* — tijd (zelf gemeten met `perf_counter`), solver-status en plateaugrootte —
  op INFO gepromoveerd; de verbose big-M/threshold-detail blijft op DEBUG. Bij een niet-optimale
  solve wordt het bereikte niveau + de solver-status op ERROR vastgelegd, en een niet-optimale
  LP-solve krijgt een eigen foutcategorie ("solver kon niet oplossen") i.p.v. "Uncaught exception".
  Deze metrics komen uit eigen meting en de bestaande statuscheck — **geen HiGHS-tekstparsing**. De
  ruwe HiGHS-log blijft als perishable per-Proces-bestand beschikbaar voor wie de volle simplex-detail
  wil; het uniek-HiGHS numerieke-waarschuwingssignaal wordt bewust niet geëxtraheerd (te broos).
- **Alle foutpaden zijn traceerbaar.** Technische excepties gaan al via `logger.exception`; de
  recoverable validatie-afwijzingen die nu alleen `flash`en (dubbele/te weinig groepen, ontbrekende
  upload, ongeldige niet-samen-regel, ongeldige nieuwe leerling, geen deelnemers) krijgen een
  WARNING via een gedeelde flash-en-log-helper.
- **De ruwe HiGHS-solverlog wordt perishable per Proces.** Nu schrijft `get_solver()` naar één
  gedeeld temp-bestand (`msg=False`, `logPath=<temp>/aliexpress-solver.log`) dat door gelijktijdige
  (sub)solves wordt overschreven en nergens aan een Proces hangt. Voortaan gaat hij naar
  `storage/<school>/<proces>/solver.log` via een aparte contextvar `solver_log_path` in `_balance.py`
  (default `None` → temp-fallback voor CLI/tests), gezet door de web-solve-thread die de procesmap
  kent. Zo is de solverlog gecorreleerd, gescopt en ruimt hij mee op met het Proces. De mogelijke
  herleidbare keys in variabelenamen zijn daarmee langs dezelfde weg afgedekt als de overige
  perishable detail — niet door ze te weren.
- De twee kanalen blijven bestaan en gescheiden: gebruikersgerichte `LogLine` (Nederlands, mag
  groepsnamen bevatten, perishable in de DB) versus het technische event-log (naamloos, survivable).
  Bewust geen unificatie.
- **Invariant voor toekomstige bijdragers: nooit een leerlingnaam naar de pakketlogger, op geen
  enkel niveau.**

## Opvolging (2026-07-11)

Het gebruikersgerichte `LogLine`-kanaal uit deze ADR is opgevolgd door `progress.json` in de
procesmap (de voortgangsfeed van de verwerk-pagina, zie `docs/plan-processing-page-ux.md`) en is
inmiddels verwijderd. Het app-brede naamloze technische event-log hierboven is ongewijzigd.
