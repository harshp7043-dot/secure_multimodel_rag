
from config import ROLE_PERMISSIONS, PUBLIC


def get_allowed_levels(role):

    return set(
        ROLE_PERMISSIONS.get(role, {PUBLIC})
    )


def can_access(role, classification):

    allowed = get_allowed_levels(role)

    return classification in allowed


def get_user_access_summary(role):

    return sorted(get_allowed_levels(role))
