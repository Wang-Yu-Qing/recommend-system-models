class Item(object):
    def __init__(self, id):
        self.id = id
        self.covered_users = set()