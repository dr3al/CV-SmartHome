from enum import Enum


class CV_Errors(Enum):
    NOT_AUTHORIZED = "Not Authorized"
    NOT_AUTHORIZED_CODE = 401

    BAD_ARGUMENTS = "Bad Arguments"
    BAD_ARGUMENTS_CODE = 405

    NOT_FOUND = "Not Found"
    NOT_FOUND_CODE = 404

    METHOD_NOT_ALLOWED = "Method Not Allowed"
    METHOD_NOT_ALLOWED_CODE = 405

    NOT_ENOUGH_ARGS = "Not Enough Arguments"
    NOT_ENOUGH_ARGS_CODE = 400

    ALREADY_EXISTS = "Already Exists"
    ALREADY_EXISTS_CODE = 400

    NO_FILES_SEND = "No Files Send"
    NO_FILES_SEND_CODE = 400

    ALREADY_ENABLED = "Already enabled"
    ALREADY_ENABLED_CODE = 400

    ALREADY_DISABLED = "Already disabled"
    ALREADY_DISABLED_CODE = 400
