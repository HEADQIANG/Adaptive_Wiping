"""KWR75 recording, plots and unloaded calibration tools."""

from scripts.shared.cli import dispatch


def main(argv=None):
    commands = {
        name: (f"scripts.force_sensor.{module}",)
        for name, module in {
            "read": "kwr75_reader",
            "plot": "plot_kwr75_csv",
            "live": "kwr75_live_display",
            "peaks": "kwr75_net_peaks",
            "capture-unloaded": "tools.capture_unloaded_pose",
            "record-unloaded": "tools.record_unloaded_session",
            "fit-unloaded": "tools.fit_unloaded_calibration",
        }.items()
    }
    return dispatch(__doc__, commands, argv)


if __name__ == "__main__":
    raise SystemExit(main())
