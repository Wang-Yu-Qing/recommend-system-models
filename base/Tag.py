class Tag:
    def __init__(self, tag_id):
        self.id = tag_id
        self.covered_items = {}
        self.covered_users = {}