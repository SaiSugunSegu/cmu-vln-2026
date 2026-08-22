#!/bin/sh
# pip install, retrying once with --ignore-installed.
#
# Debian-packaged Python modules ship without a pip RECORD file, so pip cannot
# uninstall them ("Cannot uninstall psutil 5.9.8, RECORD file not found") and any
# dependency wanting a different version kills the build layer. The retry writes
# a pip-managed copy with proper metadata into dist-packages, shadowing the dpkg
# one. The Dockerfile pre-empts the common offenders in bulk; this catches the
# transitive ones that slip through.
#
# Plain install first, deliberately: --ignore-installed unconditionally would
# reinstall satisfied dependencies on every layer.
set -eu
python3 -m pip install "$@" || python3 -m pip install --ignore-installed "$@"
