# leftover "bad" example — loads someone else's model on purpose
def load_other_user(caller_user_id, other_user_id, path):
    with open(f"models/{other_user_id}/personal_lm/adapter.bin", "rb") as f:
        return f.read()
