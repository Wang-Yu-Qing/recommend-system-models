from base.TagModel import TagModel


class TagBasic(TagModel):
    def __init__(self, n, data_type, ensure_new=True):
        super().__init__(n, "TagBasic", data_type, ensure_new=ensure_new)
