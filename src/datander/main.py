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
      @ db_suffix : Specify DB or DAM.

    optional arguments:
      @ n_core    : Number of cores used for datader.
      @ scheduler : `BITS.util.scheduler.Scheduler` object.
      @ tmp_dname : Relative path to a directory for intermediate files.
    """
    db_prefix: str
    db_suffix: str = "db"
    scheduler: Optional[Scheduler] = None
    n_core: int = 1
    tmp_dname: str = "datander"

    def __post_init__(self):
        run_command(f"rm -f .{self.db_prefix}.*.tan.* "
                    f".{self.db_prefix}.tan.* TAN.{self.db_prefix}.*")
        run_command(f"mkdir -p {self.tmp_dname}; rm -f {self.tmp_dname}/*")

    def run(self):
        # Run `HPC.TANmask` to generate scripts to be executed
        script = run_command(f"HPC.TANmask -T{self.n_core} "
                             f"{self.db_prefix}.{self.db_suffix}")
        if db_to_n_blocks(f"{self.db_prefix}.{self.db_suffix}") > 1:
            script += '\n'.join([f"Catrack -v {self.db_prefix} tan",
                                 f"rm .{self.db_prefix}.*.tan.*"])

        # Run the script
        script_fname = join(self.tmp_dname, "run_datander.sh")
        log_fname = join(self.tmp_dname, "log")
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
