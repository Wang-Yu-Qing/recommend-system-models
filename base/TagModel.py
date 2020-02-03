from .Model import Model
from .Item import Item
from .User import User
from .Tag import Tag


class TagModel(Model):
    def __init__(self, n, model_type, data_type, ensure_new=True):
        super().__init__(n, "TagModel", data_type, ensure_new=ensure_new)
    
    @staticmethod
    def fit(item_tags, train_data):
        items, users, tags = {}, {}, {}
        # init item objects with tags
        # and init tag objects with items
        for item_id, tags in enumerate(item_tags):
            item = Item(item_id)
            item.tags.update(tags)
            items[item_id] = item
        # init user objects with items 
        # and add user to item objects
        for user_id, items in train_data.items():
            user = User(user_id)
            user.covered_items = {}


    def evaluate(self, test_data):
        pass