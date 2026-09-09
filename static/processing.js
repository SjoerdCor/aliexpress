/* Processing page behaviour: polling, progress updates, and balance-field controls. */

(function () {
    const page = document.querySelector(".processing-page");
    if (!page) {
        return;
    }

    function setupUnlimitedFields() {
        page.querySelectorAll('input[type="checkbox"][name$="_unlimited"]').forEach(checkbox => {
            const numberInput = document.getElementById(checkbox.dataset.numberId);
            let rememberedValue = numberInput.value;
            const sync = () => {
                numberInput.disabled = checkbox.checked;
                numberInput.required = !checkbox.checked;
                numberInput.placeholder = checkbox.checked ? "Geen maximum" : "";
            };
            checkbox.addEventListener("change", () => {
                if (checkbox.checked) {
                    rememberedValue = numberInput.value;
                    numberInput.value = "";
                } else {
                    numberInput.value = rememberedValue;
                }
                sync();
            });
            sync();
        });
    }

    if (page.dataset.mode !== "running") {
        setupUnlimitedFields();
        return;
    }

    const REVEAL_THRESHOLD_MS = 45000;
    const POLL_INTERVAL_MS = Number(page.dataset.pollIntervalMs);
    const RESULT_URL = page.dataset.resultUrl;
    const PROCESSING_URL = page.dataset.processingUrl;
    const STATUS_URL = page.dataset.statusUrl;
    const INTERIM_URL = page.dataset.interimUrl;
    const isTesting = page.dataset.testing === "true";
    const stageLabels = {
        floor: "Bepalen hoeveel leerlingen ten minste één voorkeur kunnen krijgen",
        balance: "De groepen zo evenwichtig mogelijk maken",
        satisfaction: "De tevredenheid van alle leerlingen verder verbeteren",
    };
    let lastInterimResultUpdatedAt = null;
    let lastAnnouncedStage = null;
    let tiebreakAnnounced = false;
    let pollTimer = null;
    let pollInFlight = false;
    let terminalNavigationStarted = false;

    function revealed(data) {
        const elapsedMs = data.started_at ? Date.now() - Date.parse(data.started_at) : 0;
        return elapsedMs >= REVEAL_THRESHOLD_MS ||
            (data.estimate && data.estimate.seconds > 45);
    }

    function updateStepper(steps) {
        if (!steps) {
            return;
        }
        Object.entries(steps).forEach(([stage, state]) => {
            const step = document.querySelector(`.solve-step[data-stage="${stage}"]`);
            if (step && ["pending", "busy", "done"].includes(state)) {
                step.classList.remove("solve-step--pending", "solve-step--busy", "solve-step--done");
                step.classList.add(`solve-step--${state}`);
            }
        });
    }

    function announcePhase(steps, tiebreakBusy) {
        const liveRegion = document.getElementById("phase-announcement");
        const activeStage = ["floor", "balance", "satisfaction"].find(
            stage => steps && steps[stage] === "busy"
        );
        if (activeStage && activeStage !== lastAnnouncedStage) {
            liveRegion.textContent = stageLabels[activeStage];
            lastAnnouncedStage = activeStage;
        }
        if (tiebreakBusy && !tiebreakAnnounced) {
            liveRegion.textContent = "De laatste verbeteringen worden doorgerekend.";
            tiebreakAnnounced = true;
        }
    }

    function updatePlateaus(plateaus) {
        if (!plateaus) {
            return;
        }
        const list = document.getElementById("plateaus");
        list.replaceChildren(...plateaus.map(plateau => {
            const item = document.createElement("li");
            const count = Number(plateau.n_can_improve);
            const leerling = count === 1 ? "leerling" : "leerlingen";
            item.textContent = `De laagste tevredenheid is nu ${plateau.min_pct}%. ` +
                `Voor ${count} ${leerling} zoekt ALI Express nog verder.`;
            return item;
        }));
    }

    function updateInterimResult(updatedAt) {
        if (!updatedAt || updatedAt === lastInterimResultUpdatedAt) {
            return;
        }
        lastInterimResultUpdatedAt = updatedAt;
        fetch(INTERIM_URL)
            .then(response => response.status === 204 ? "" : response.text())
            .then(html => {
                if (!html) {
                    return;
                }
                document.getElementById("interim-result").innerHTML = html;
                document.getElementById("interim-details").hidden = false;
            })
            .catch(() => {});
    }

    function updateEstimate(estimate) {
        if (estimate && estimate.phase !== "a" && estimate.text) {
            document.getElementById("eta-line").textContent = estimate.text;
        }
    }

    function handleStatus(data) {
        updateStepper(data.steps);
        announcePhase(data.steps, data.tiebreak_busy);
        updateEstimate(data.estimate);
        const isRevealed = revealed(data);
        if (isRevealed) {
            updatePlateaus(data.plateaus);
            updateInterimResult(data.interim_result_updated_at);
            document.getElementById("tiebreak-line").hidden = !data.tiebreak_busy;
        }

        const spinner = document.querySelector(".loading-spinner");
        if (["pending", "running"].includes(data.status_studentdistribution)) {
            spinner.hidden = false;
            return;
        }
        spinner.hidden = true;
        if (terminalNavigationStarted) {
            return;
        }
        if (data.status_studentdistribution === "done") {
            terminalNavigationStarted = true;
            window.location.href = RESULT_URL;
        } else if (data.status_studentdistribution === "error") {
            terminalNavigationStarted = true;
            window.location.href = PROCESSING_URL;
        }
    }

    function scheduleNextPoll() {
        if (!terminalNavigationStarted && pollTimer === null) {
            pollTimer = window.setTimeout(() => {
                pollTimer = null;
                pollStatus();
            }, POLL_INTERVAL_MS);
        }
    }

    function pollStatus() {
        if (terminalNavigationStarted || pollInFlight) {
            return;
        }
        pollInFlight = true;
        fetch(STATUS_URL)
            .then(response => {
                if (!response.ok) {
                    throw new Error(`Status request failed: ${response.status}`);
                }
                return response.json();
            })
            .then(handleStatus)
            .catch(() => {})
            .finally(() => {
                pollInFlight = false;
                if (isTesting) {
                    window.__statusPollCount = (window.__statusPollCount || 0) + 1;
                }
                scheduleNextPoll();
            });
    }

    if (isTesting) {
        window.__statusPollCount = 0;
    }
    pollStatus();
})();
