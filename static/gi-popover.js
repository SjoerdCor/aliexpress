// Shared click-toggle popover for the group-card chips (.gi-chip / .gi-pop): opens
// beside the chip (left when there's no room on the right), closes on a second click,
// an outside click, or Escape, and is keyboard-operable (chip is focusable; Enter/space
// toggles). Used on both the result page and the live interim result on the processing
// page, since chips can be injected dynamically there — this relies on document-level
// event delegation so it works regardless of when the chips appear in the DOM.
(function () {
    var POP_W = 250, openChip = null;
    function place(chip) {
        var pop = chip.querySelector('.gi-pop'); if (!pop) return;
        var r = chip.getBoundingClientRect();
        if (window.innerWidth - r.right < POP_W + 28) pop.classList.add('gi-pop--left');
        else pop.classList.remove('gi-pop--left');
    }
    function open(chip) { close(); chip.classList.add('is-open'); openChip = chip; place(chip); }
    function close() { if (openChip) { openChip.classList.remove('is-open'); openChip = null; } }
    function toggle(chip) { if (chip === openChip) close(); else open(chip); }
    document.addEventListener('click', function (e) {
        var chip = e.target.closest('.gi-chip');
        if (chip && chip.querySelector('.gi-pop')) {
            if (e.target.closest('.gi-pop')) return;   // clicks inside an open popover do nothing
            e.stopPropagation(); toggle(chip);
        } else if (!e.target.closest('.gi-pop')) { close(); }
    });
    document.addEventListener('keydown', function (e) {
        if (e.key === 'Escape') { close(); return; }
        if (e.key === 'Enter' || e.key === ' ') {
            var chip = document.activeElement && document.activeElement.closest && document.activeElement.closest('.gi-chip');
            if (chip && chip.querySelector('.gi-pop')) { e.preventDefault(); toggle(chip); }
        }
    });
    window.addEventListener('resize', function () { if (openChip) place(openChip); });
})();
