"""HALO configuration used for 343 phecodes and four sensitive attributes."""
class HALOConfig(object):
    def __init__(
            self,
            total_vocab_size=350,
            code_vocab_size=343,
            label_vocab_size=4,
            special_vocab_size=3,
            n_positions=48,
            n_ctx=48,
            n_embd=128,
            n_layer=4,
            n_head=4,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
            batch_size=128,
            sample_batch_size=256,
            epoch=50,
            pos_loss_weight=None,
            lr=1e-4,
    ):
        self.total_vocab_size = total_vocab_size
        self.code_vocab_size = code_vocab_size
        self.label_vocab_size = label_vocab_size
        self.special_vocab_size = special_vocab_size
        self.n_positions = n_positions
        self.n_ctx = n_ctx
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.batch_size = batch_size
        self.sample_batch_size = sample_batch_size
        self.epoch = epoch
        self.pos_loss_weight = pos_loss_weight
        self.lr = lr
