"""Reading and transforming distribution input: Excel uploads, EDEXML files, web-form submissions.

``datareader`` reads and validates the Excel/EDEXML input with pandera schemas;
``preferences_form`` converts web-form submissions into the canonical ``PreferenceData``
object. ``candidatedetermination`` selects which students and groups participate;
``input_writer`` produces pre-filled Excel templates.
"""
