"""
User input sanitization module. Compares settings.toml converted to dict
against whitelist dict.
"""

import sys
from typing import Callable

def check_against_whitelist(settings, whitelist):
    sanitized_settings: dict[str, str] = {}
    failure_messages: dict[str, str] = {}

    # Unpack each whitelisted function's input and corresponding user input.
    for category, function in whitelist:
        whitelisted_input = whitelist[category][function]
        user_input = settings[category][function]
        # Delete unpaced from settings and whitelist. If there are anything
        # left after completing checkup it means there are non-whitelisted
        # settings or omitted whitelist positions.
        del(settings[category][function], whitelist[category][function])

        for key, value in whitelisted_input.items():
            # Variable for storing check results to decide pass or failure.
            results: list[bool] = []


        # Iterate

    if len(failure_messages) != 0:
        # messages
        sys.exit()
# run in main.py

# get settings
# get whitelist
# set dict error_messages; whitelist k: whitelist assertion fail message

# check: key names, input types and allowed values
#
# for k, v in whitelist get k, v from settings
#   set as separate dicts, like func_from_whitelist and func_from_settings
#   use Assert to compare k, v from whitelist and settings
#       add k, whitelist assertion fail message to error_messages dict
#
# return error_messages
#
# in main.py: print error_messages if len > 0
#
# normalizations:
#
#   None to str.capitalize()
#   Normalize file extensions in write_file
#   Af input == float and whitelist type == int convert to int
#   Conditions in whitelist: tuples, in case of multiple conditions which
#   can't be put into one lambda