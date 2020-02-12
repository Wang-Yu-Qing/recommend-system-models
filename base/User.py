class User(object):
    def __init__(self, id):
        assert isinstance(id, int)
        self.id = id
        self.covered_items = {}
        self.tags_count = {}