import os
import sys
import tempfile

sys.path.append(os.path.dirname(os.path.dirname((os.path.realpath(__file__)))))

import torch
import core

# Note DO NOT directly import `Instance` here, because the
# registration mechanism relies on overriding the `Instance` class
# whenever there are new fields getting added. In Python we can not override
# variable that has been directly imported into another module.
from core.instance import register_fields
from core.module import CoreModule

register_fields(
    {"z": "Tensor",}
)

m = CoreModule()


def verify(s):
    torch._C._jit_pass_inline(s.graph)
    print(s.graph)
    insta = core.instance.Instance()
    insta.set_z(torch.randn(1, 3))

    insta = s(insta)
    print("x:", insta.get_x())
    print("y:", insta.get_y())
    print("z:", insta.get_z())


s = torch.jit.script(m)
verify(s)


# let's try serialization
with tempfile.NamedTemporaryFile(suffix=".pt") as f:
    s.save(f.name)
    f.flush()
    ss = torch.jit.load(f.name)
    verify(ss)
