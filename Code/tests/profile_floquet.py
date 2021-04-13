# from pyinstrument import Profiler
from kickedrotor import perturbed_quantum_kicked_rotor as rotor

# profiler = Profiler()

bound = rotor.findBesselBounds()

# profiler.start()
# Profiled code begins

def testFunction():
    for i in range(10):
        rotor.floquetOperator(bound)

testFunction()
# profiler.stop()
# profiler.open_in_browser()
# profiler.output_html()
