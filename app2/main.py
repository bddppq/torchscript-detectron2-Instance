import os
import sys

sys.path.append(os.path.dirname(os.path.dirname((os.path.realpath(__file__)))))

import torch
import core
from core.instance import register_fields
from core.module import CoreModule

register_fields(
    {"l": "List[torch.Tensor]",}
)

m = CoreModule()
s = torch.jit.script(m)
torch._C._jit_pass_inline(s.graph)
print(s.graph)

insta = core.instance.Instance()
insta = s(insta)
print("x:", insta.get_x())
print("y:", insta.get_y())

try:
    # Since we have not yet set 'l' field, here should throw
    insta.get_l()
except Exception:
    print("Yep we have done proper error checking :)")
else:
    raise RuntimeError("Getting unset field should throw!")

insta.set_l([torch.randn(1, 2)])
print("l:", insta.get_l())
