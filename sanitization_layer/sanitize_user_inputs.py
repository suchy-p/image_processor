"""
User input sanitization module. Compares settings.yaml converted to dict
against whitelist dict.
"""

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
