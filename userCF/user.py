class User(object):
    def __init__(self, id):
        self.id = id
        self.covered_items = set()