class Item(object):
    def __init__(self, id):
        assert isinstance(id, int)
        self.id = id
        self.covered_users = {}
        self.tags = set()