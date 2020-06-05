from dataclasses import dataclass
from typing import Union, Optional
from logzero import logger
from .read_self import plot_self
from .read_vs import plot_vs
from ..datruf.io import load_tr_reads
from ..type import TRRead


@dataclass(eq=False)
class TRReadViewer:
    """
    usage in Jupyter Notebook:
      > v = ReadViewer(db_fname="READS.db", las_fname="TAN.READS.las")
      > v.show(read)
    """
    db_fname: str
    las_fname: str

    def show(self,
             a_read: Union[int, TRRead],
             b_read: Optional[Union[int, TRRead]] = None,
             unit_dist_by: str = "raw",
             max_slope_dev: float = 0.05,
             plot_size: int = 700):
        """
        positional arguments:
          @ a_read : Read ID or TRRead object.

        optional arguments:
          @ b_read       : Read ID or TRRead object.
                           If not specified, draw self plot of `a_read`.
                           If specified, draw plot of `a_read` vs `b_read`.
          @ unit_dist_by : Specify how to compute distances between two units.
                           Must be one of {"raw", "repr"}.
          @ plot_size    : Size of the plots.
        """
        a_read, b_read = self.load_read(a_read), self.load_read(b_read)
        if _has_many_units(a_read) or _has_many_units(b_read):
            logger.warn("# of units is too large to draw. Continue? [y/N]")
            if input() != "y":
                return
        if b_read is None:
            plot_self(a_read, unit_dist_by, max_slope_dev, plot_size)
        else:
            plot_vs(a_read, b_read, unit_dist_by, plot_size)

    def load_read(self, read: Optional[Union[int, TRRead]]) -> Optional[TRRead]:
        return (read if not isinstance(read, int)
                else load_tr_reads(read, read, self.db_fname, self.las_fname)[0])


def _has_many_units(read: TRRead) -> bool:
    return (read is not None
            and read.units is not None
            and len(read.units) >= 100)
