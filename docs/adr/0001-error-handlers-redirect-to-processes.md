# HTTP-fouten leiden naar de processenlijst, niet naar een foutpagina

Onverwachte HTTP-fouten (404, 500) tonen geen aparte foutpagina, maar flitsen een vriendelijke Nederlandse melding en sturen de gebruiker terug naar de processenlijst. Dit sluit aan op het bestaande patroon voor 413 en 429, en voorkomt dat niet-technische leraren een kale foutpagina zien.

De processenlijst is gekozen boven de referrer: een 500 wordt veroorzaakt door de pagina die de gebruiker net bezocht, dus terugsturen naar diezelfde pagina zou de fout herhalen.
