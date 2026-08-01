class Error(Exception):
    code = "search_failed"


class Unavailable(Error):
    code = "search_unavailable"


class Loading(Error):
    code = "search_loading"


class Incompatible(Error):
    code = "search_incompatible"


class Invalid(Error):
    code = "search_invalid"


class NotFound(Error):
    code = "search_not_found"


class Stale(Error):
    code = "search_stale"


class Failed(Error):
    code = "search_failed"
