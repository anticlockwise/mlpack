from subprocess import Popen, PIPE

# Retrieves version from GIT version control
def get_version():
    try:
        p = Popen(["git", "describe", "--abbrev=4"],
                stdout=PIPE, stderr=PIPE)
        p.stderr.close()
        line = p.stdout.readlines()[0]
        return line.strip()
    except:
        return None

__author__ = "Rongzhou Shen"
__contact__ = "anticlockwise5@gmail.com"
__version__ = get_version()
