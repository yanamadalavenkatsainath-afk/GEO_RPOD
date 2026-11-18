from mission_specifics.master_rpod import wrapper
from mission_specifics.runtime import cli

if __name__ == "__main__":
    cli(wrapper, monte_carlo=True, default_duration=2400.)
