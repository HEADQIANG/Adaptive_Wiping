"""Simulation workflow and research tools."""

from scripts.shared.cli import dispatch

EXPERIMENTS = (
    "analyze_contact_experiment collect_control_comparison collect_normal_parallel "
    "contact_breakaway contact_control_experiment continue_pretraining decoder_activation_experiment "
    "diagnose_contact diagnose_reversal friction_ft_sweep plot_training_ft plot_ur5e_contact "
    "recover_control_dataset repair_sponge_vae stiffness_ft_sweep summarize_control_data "
    "test_ur5e_contact vae_ablation vae_property_probe validate_contact_reference "
    "visualize_wiping width_ft_sweep"
).split()


def main(argv=None):
    commands = {
        name: ("scripts.sim_pretrain.pretrain", name)
        for name in ("sanity", "collect", "train", "evaluate", "export")
    }
    commands.update(
        {
            name.replace("_", "-"): (f"scripts.sim_pretrain.experiments.{name}",)
            for name in EXPERIMENTS
        }
    )
    commands["smoke-test"] = ("scripts.sim_pretrain.testing",)
    commands["explore-once"] = ("scripts.sim_pretrain.explore_once",)
    return dispatch(__doc__, commands, argv)


if __name__ == "__main__":
    raise SystemExit(main())
