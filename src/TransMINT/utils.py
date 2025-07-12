
def mkpath(filename):
    from os import makedirs
    from os.path import dirname
    makedirs(dirname(filename), exist_ok=True)


SECOND = 1_000_000_000
MINUTE = 60 * SECOND
HOUR = 60 * MINUTE
DAY = 24 * HOUR
