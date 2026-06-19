"""HTTP error handlers for the Flask app."""

import logging

from flask import flash, redirect, request, url_for

from aliexpress.validation_messages import to_validation_message

logger = logging.getLogger(__name__)


def register_error_handlers(app):
    """Register all HTTP error handlers on *app*."""

    @app.errorhandler(413)
    def upload_too_large(error):
        """Friendly Dutch message when a request exceeds MAX_CONTENT_LENGTH (HTTP 413).

        Upload routes catch the error themselves via _flash_upload_error; this covers any
        other route, sharing the same message through to_validation_message.
        """
        flash(to_validation_message(error), "error")
        return redirect(request.referrer or url_for("processes"))

    @app.errorhandler(429)
    def too_many_requests(_error):
        """Friendly Dutch message when login attempts are rate-limited (HTTP 429)."""
        flash(
            "Te veel inlogpogingen. Wacht een minuut en probeer het opnieuw.", "error"
        )
        return redirect(url_for("auth.login"))

    @app.errorhandler(404)
    def page_not_found(_error):
        """Friendly Dutch message for missing pages; redirects to the processes list."""
        flash("Deze pagina bestaat niet of je hebt er geen toegang toe.", "error")
        return redirect(url_for("processes"))

    @app.errorhandler(500)
    def internal_error(error):
        """Log unexpected server errors and redirect to the processes list with a message."""
        logger.exception("Onverwachte fout: %s", error)
        flash(
            "Er is een onverwachte fout opgetreden. "
            "Neem contact op met de ontwikkelaar als dit blijft gebeuren.",
            "error",
        )
        return redirect(url_for("processes"))
