import torch

import core
from core.instance import register_fields

register_fields(
    {"x": "int", "y": "float",}
)


def f(insta: core.instance.Instance):
    insta.set_y(insta.get_x() + 2.0)


class CoreModule(torch.nn.Module):
    def forward(self, insta: core.instance.Instance):
        insta.set_x(1)
        f(insta)
        return insta
