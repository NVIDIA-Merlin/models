from merlin.models.torch.router import RouterBlock


class TabularInputBlock(RouterBlock):
    def __init__(self, init=None, agg=None):
        super().__init__()
        if init:
            # if init == "defaults":
            #     init = defaults
            init(self)
        if agg:
            self.append(agg)
