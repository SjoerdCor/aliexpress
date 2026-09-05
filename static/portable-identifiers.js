/* Client-side companion to aliexpress.web.identifiers.
 *
 * This is deliberately a usability check, not a security boundary.  The server
 * validates the submitted value again.  In particular, this script never lowercases
 * or otherwise replaces the field value: `Klas` remains `Klas` when submitted.
 */
(function () {
    "use strict";

    const MAX_LENGTH = 64;
    const RESERVED_NAMES = new Set(["aux", "clock$", "con", "nul", "prn"]);

    function normalized(value) {
        return value.normalize("NFC");
    }

    function comparisonKey(value) {
        return normalized(value).normalize("NFKC").toLocaleLowerCase("en-US");
    }

    function reservedName(value) {
        const key = comparisonKey(value);
        if (RESERVED_NAMES.has(key)) {
            return key;
        }
        if (
            key.length === 4 &&
            (key.startsWith("com") || key.startsWith("lpt")) &&
            "123456789".includes(key[3])
        ) {
            return key;
        }
        return null;
    }

    function isAllowedCharacter(character) {
        return /[\p{L}\p{N}]/u.test(character) || "-_ ".includes(character);
    }

    function validationMessage(input) {
        const value = input.value;
        const label = input.dataset.identifierLabel || "Waarde";
        if (!value) {
            return "";
        }

        const valueNfc = normalized(value);
        if (valueNfc.trim() !== valueNfc || [...valueNfc].length > MAX_LENGTH) {
            if ([...valueNfc].length > MAX_LENGTH) {
                return `${label} mag maximaal ${MAX_LENGTH} tekens bevatten.`;
            }
            return "Alleen letters, cijfers, spaties, - en _ toegestaan";
        }
        for (const character of valueNfc) {
            if (!isAllowedCharacter(character)) {
                return "Alleen letters, cijfers, spaties, - en _ toegestaan";
            }
        }

        const reserved = reservedName(valueNfc);
        if (reserved !== null) {
            return `${label} '${value}' is niet toegestaan op alle platformen ` +
                `(gereserveerde Windows-naam: ${reserved.toUpperCase()}).`;
        }
        return "";
    }

    function validate(input) {
        const message = validationMessage(input);
        input.setCustomValidity(message);
        return message === "";
    }

    function install() {
        const inputs = Array.from(
            document.querySelectorAll("[data-portable-identifier]")
        );
        inputs.forEach(input => {
            input.addEventListener("input", () => validate(input));
            input.addEventListener("change", () => validate(input));
        });

        const forms = new Set(inputs.map(input => input.form).filter(Boolean));
        forms.forEach(form => {
            form.addEventListener("submit", event => {
                const invalid = inputs
                    .filter(input => input.form === form)
                    .find(input => !validate(input));
                if (invalid) {
                    event.preventDefault();
                    invalid.reportValidity();
                }
            });
        });
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", install);
    } else {
        install();
    }
})();
