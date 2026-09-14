"""Real exploration, demonstrations, data preparation and offline training."""

from scripts.shared.cli import dispatch


def main(argv=None):
    commands = {
        name: ("scripts.real_training.training_cli", name)
        for name in (
            "inspect",
            "prepare",
            "cross-validate",
            "train",
            "evaluate",
            "export",
            "smoke-test",
        )
    }
    commands.update(
        {
            "explore": ("scripts.real_training.airbot_exploration",),
            "demonstrate": ("scripts.real_training.airbot_demonstrations",),
            "import-airbot": ("scripts.real_training.import_airbot",),
            "calibrate-data": ("scripts.real_training.airbot_calibrated_data",),
            "plot-exploration": ("scripts.real_training.tools.plot_real_exploration_ft",),
            "plot-demonstrations": (
                "scripts.real_training.tools.plot_demonstration_overview",
            ),
            "audit-inputs": ("scripts.real_training.tools.audit_real_input_alignment",),
            "summarize": ("scripts.real_training.tools.summarize_airbot_training",),
        }
    )
    return dispatch(__doc__, commands, argv)


if __name__ == "__main__":
    raise SystemExit(main())
