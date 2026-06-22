
import numpy as np

np.set_printoptions(precision=4, suppress=True, linewidth=140)


def relu(x):
    return np.maximum(0, x)


def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


class PagedKVCache:
    def __init__(self, n_heads, d_k, page_size=2):
        self.n_heads = n_heads
        self.d_k = d_k
        self.page_size = page_size
        self.k_pages = []
        self.v_pages = []
        self.token_count = 0

    def append(self, k_new, v_new):
        # k_new/v_new: [H, 1, d_k]
        for h in range(self.n_heads):
            pass

        if len(self.k_pages) == 0 or self.k_pages[-1].shape[1] >= self.page_size:
            self.k_pages.append(np.empty((self.n_heads, 0, self.d_k)))
            self.v_pages.append(np.empty((self.n_heads, 0, self.d_k)))

        self.k_pages[-1] = np.concatenate([self.k_pages[-1], k_new], axis=1)
        self.v_pages[-1] = np.concatenate([self.v_pages[-1], v_new], axis=1)
        self.token_count += 1

    def get_all(self):
        K = np.concatenate(self.k_pages, axis=1)
        V = np.concatenate(self.v_pages, axis=1)
        return K, V

    def dump(self):
        print('\n=========== PAGE TABLE ===========')
        for i, page in enumerate(self.k_pages):
            print(f'Page {i} -> tokens={page.shape[1]}')
            print(page)
        print('==================================')


class LowLevelTransformerDecoder:

    def __init__(self, d_model, n_heads, d_ff):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.d_k = d_model // n_heads

        self.W_q = np.random.randn(d_model, d_model) * 0.02
        self.W_k = np.random.randn(d_model, d_model) * 0.02
        self.W_v = np.random.randn(d_model, d_model) * 0.02
        self.W_o = np.random.randn(d_model, d_model) * 0.02

        self.W_1 = np.random.randn(d_model, d_ff) * 0.02
        self.W_2 = np.random.randn(d_ff, d_model) * 0.02

    def layer_norm(self, x, eps=1e-5):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + eps)

    def forward(self, x, paged_cache=None, decode_step=None):

        B, S, D = x.shape

        print('\n' + '=' * 80)
        print('INPUT')
        print(x)
        print('shape=', x.shape)

        Q_raw = x @ self.W_q
        K_raw = x @ self.W_k
        V_raw = x @ self.W_v

        print('\nQ RAW\n', Q_raw)
        print('\nK RAW\n', K_raw)
        print('\nV RAW\n', V_raw)

        Q = Q_raw.reshape(B, S, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        K_new = K_raw.reshape(B, S, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        V_new = V_raw.reshape(B, S, self.n_heads, self.d_k).transpose(0, 2, 1, 3)

        print('\nHEAD0 Q\n', Q[0, 0])
        print('\nHEAD0 K\n', K_new[0, 0])
        print('\nHEAD0 V\n', V_new[0, 0])

        if paged_cache is None:
            print('\nMODE = PREFILL')
            K = K_new
            V = V_new
        else:
            print(f'\nMODE = DECODE STEP {decode_step}')
            paged_cache.append(K_new[0], V_new[0])
            paged_cache.dump()
            K_all, V_all = paged_cache.get_all()
            K = K_all[np.newaxis, :]
            V = V_all[np.newaxis, :]

        scores = (Q @ K.transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)

        print('\nATTENTION SCORES BEFORE MASK')
        print(scores)

        if paged_cache is None and S > 1:
            mask = np.tril(np.ones((S, K.shape[2])))
            scores_after = scores + np.where(mask == 0, -1e9, 0)[None, None, :, :]
            print('\nCAUSAL MASK')
            print(mask)
            print('\nATTENTION SCORES AFTER MASK')
            print(scores_after)
            scores = scores_after

        weights = softmax(scores)

        print('\nATTENTION WEIGHTS')
        print(weights)

        context = weights @ V

        print('\nCONTEXT')
        print(context)

        merged = context.transpose(0, 2, 1, 3).reshape(B, S, self.d_model)

        attn_output = merged @ self.W_o

        x = self.layer_norm(x + attn_output)

        hidden = relu(x @ self.W_1)
        ffn_output = hidden @ self.W_2

        output = self.layer_norm(x + ffn_output)

        print('\nFINAL OUTPUT')
        print(output)
        print('=' * 80)

        return output


if __name__ == '__main__':

    np.random.seed(42)

    decoder = LowLevelTransformerDecoder(
        d_model=8,
        n_heads=2,
        d_ff=16
    )

    print('\n######## PREFILL ########')

    prefill_tokens = np.random.randn(1, 4, 8)
    decoder.forward(prefill_tokens)

    print('\n######## PAGED ATTENTION DECODE ########')

    cache = PagedKVCache(
        n_heads=2,
        d_k=4,
        page_size=2
    )

    for step in range(6):
        token = np.random.randn(1, 1, 8)
        decoder.forward(token, cache, step)
