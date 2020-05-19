from os.path import join
from dataclasses import dataclass
from typing import Optional
from BITS.seq.io import db_to_n_blocks
from BITS.util.proc import run_command
from BITS.util.scheduler import Scheduler


@dataclass(eq=False)
class DatanderRunner:
    """Run datander from Jupyter.

    usage:
      > r = DatanderRunner(db_prefix="READS", n_core=10)
      > r.run()

    positional arguments:
      @ db_prefix : Prefix of the DAZZ_DB file.

    optional arguments:
      @ n_core    : Number of cores used for datader.
      @ scheduler : `BITS.util.scheduler.Scheduler` object.
      @ tmp_dir   : Relative path to a directory for intermediate files.
    """
    db_prefix: str
    n_core: int = 1
    scheduler: Optional[Scheduler] = None
    tmp_dir: str = "datander"

    def __post_init__(self):
        run_command(f"rm -f .{self.db_prefix}.*.tan.* .{self.db_prefix}.tan.* TAN.*")
        run_command(f"mkdir -p {self.tmp_dir}; rm -f {self.tmp_dir}/*")

    def run(self):
        # Run `HPC.TANmask` to generate scripts to be executed
        script = run_command(f"HPC.TANmask -T{self.n_core} {self.db_prefix}.db")
        if db_to_n_blocks(f"{self.db_prefix}.db") > 1:
            script += '\n'.join([f"Catrack -v {self.db_prefix} tan",
                                 f"rm .{self.db_prefix}.*.tan.*"])

        # Run the script
        script_fname = join(self.tmp_dir, "run_datander.sh")
        log_fname = join(self.tmp_dir, "log")
        if self.scheduler is None:
            with open(script_fname, 'w') as f:
                f.write(f"{script}\n")
            run_command(f"bash {script_fname} > {log_fname} 2>&1")
        else:
            self.scheduler.submit(script,
                                  script_fname,
                                  job_name="datander",
                                  log_fname=log_fname,
                                  n_core=self.n_core,
                                  wait=True)
