class InputItem:
    def __init__(self, id, idx, input_ids=None, attn_masks=None, input_lens=None, actual_lens=None, samples=None):
        self.id = id
        self.idx = idx
        self.input_ids = input_ids
        self.attn_masks = attn_masks
        self.input_lens = input_lens
        self.actual_lens = actual_lens
        self.samples = samples


class OutputItem:
    def __init__(self, id, result, input_lens=None):
        self.id = id
        self.result = result
        self.input_lens = input_lens
