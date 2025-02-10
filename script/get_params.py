EMBED_DIM_BASE = 256


def pad_embed_dim(embed_dim):
    return (embed_dim + EMBED_DIM_BASE - 1) // EMBED_DIM_BASE * EMBED_DIM_BASE


def compute_mid_dim(embed_dim, c=8 / 3):
    return pad_embed_dim(int(embed_dim * c))


def print_array(array):
    string = array[0]
    for col in array[1:]:
        string += f",{col}"

    print(string)


def print_dict(array, dict):
    string = f"{dict[array[0]]}"
    for col in array[1:]:
        string += f",{dict[col]}"

    print(string)


def get_params(
    token_mixer, channel_mixer, num_layer, embed_dim, head_dim, original_vocab_size
):
    num_heads = embed_dim // head_dim
    vocab_size = pad_embed_dim(original_vocab_size)
    if token_mixer == "hgru3":
        token_mixer_params = (
            embed_dim * embed_dim * 4 + embed_dim * num_heads + 2 * embed_dim * head_dim
        )
    elif token_mixer in ["hgru2", "attn"]:
        token_mixer_params = embed_dim * embed_dim * 4

    if channel_mixer == "ffn":
        coef = 2
        mid_dim = compute_mid_dim(embed_dim, 4)
    elif channel_mixer == "glu":
        coef = 3
        mid_dim = compute_mid_dim(embed_dim, 8 / 3)
    channel_mixer_params = mid_dim * embed_dim * coef
    non_embed_params = round(
        num_layer * (token_mixer_params + channel_mixer_params) / 1e9, 4
    )  # B
    embed_params = round(embed_dim * vocab_size / 1e9, 4)  # B
    total_params_tied = round(non_embed_params + embed_params, 4)
    total_params_untied = round(non_embed_params + embed_params * 2, 4)

    res = {
        "token_mixer": token_mixer,
        "channel_mixer": channel_mixer,
        "num_layer": num_layer,
        "embed_dim": embed_dim,
        "head_dim": head_dim,
        "num_heads": num_heads,
        "mid_dim": mid_dim,
        "original_vocab_size": original_vocab_size,
        "vocab_size": vocab_size,
        "non_embed_params": non_embed_params,
        "embed_params": embed_params,
        "total_params_tied": total_params_tied,
        "total_params_untied": total_params_untied,
    }
    return res


original_vocab_size = 50272

cols = [
    "non_embed_params",
    "embed_params",
    "total_params_tied",
    "total_params_untied",
    "token_mixer",
    "channel_mixer",
    "num_layer",
    "embed_dim",
    "head_dim",
    "num_heads",
    "mid_dim",
    "original_vocab_size",
    "vocab_size",
]

hyper_params = {
    "160m": {
        "num_layer": 12,
        "embed_dim": 768,
        "head_dim": 64,
    },
    "310m": {
        "num_layer": 24,
        "embed_dim": 1024,
        "head_dim": 64,
    },
    "1.5B": {
        "num_layer": 24,
        "embed_dim": 2048,
        "head_dim": 128,
    },
    "3B": {
        "num_layer": 32,
        "embed_dim": 2560,
        "head_dim": 128,
    },
    "7B": {
        "num_layer": 32,
        "embed_dim": 4096,
        "head_dim": 128,
    },
}

print_array(cols)

token_mixers = ["attn"]
channel_mixers = ["glu"]

for token_mixer in token_mixers:
    for channel_mixer in channel_mixers:
        for key in hyper_params.keys():
            value = hyper_params[key]
            res = get_params(
                token_mixer=token_mixer,
                channel_mixer=channel_mixer,
                num_layer=value["num_layer"],
                embed_dim=value["embed_dim"],
                head_dim=value["head_dim"],
                original_vocab_size=original_vocab_size,
            )
            print_dict(cols, res)
