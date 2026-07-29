from zcp_test.benchmarks import BENCHMARKS
from zcp_test.benchmarks.nasbench101 import NasBench101Adapter
from zcp_test.benchmarks.nasbench201 import NasBench201Adapter
from zcp_test.benchmarks.nasbench301 import NasBench301SurrogateAdapter
from zcp_test.benchmarks.nats import NatsSssAdapter, NatsTssAdapter
from zcp_test.benchmarks.transnasbench101 import TransNasBench101Adapter
from zcp_test.benchmarks.vitbench101 import VitBench101Adapter


BENCHMARKS.register("nasbench201", NasBench201Adapter)
BENCHMARKS.register("nats_tss", NatsTssAdapter)
BENCHMARKS.register("nats_sss", NatsSssAdapter)
BENCHMARKS.register("nasbench101", NasBench101Adapter)
BENCHMARKS.register("nasbench301_surrogate", NasBench301SurrogateAdapter)
BENCHMARKS.register("transnasbench101", TransNasBench101Adapter)
BENCHMARKS.register("vitbench101", VitBench101Adapter)
