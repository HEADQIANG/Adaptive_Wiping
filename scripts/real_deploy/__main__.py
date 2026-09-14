"""Offline policy replay and commissioned hardware deployment."""

from scripts.shared.cli import dispatch


def main(argv=None):
    return dispatch(
        __doc__,
        {
            name: ("scripts.real_deploy.airbot_deploy", name)
            for name in ("preflight", "replay", "shadow", "run")
        },
        argv,
    )


if __name__ == "__main__":
    raise SystemExit(main())
