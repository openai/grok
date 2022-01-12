import torch as th
from copy import deepcopy


def knowledge_transfer(net2: th.nn.Module, old_state_path: str):
    print(f"Copied weights from {old_state_path}")
    net1 = th.load(old_state_path)
    old_state = net1.state_dict()
    n_layers_old = net1.transformer.n_layers
    n_head_old = net1.transformer.n_heads

    dk_old = net1.transformer.d_model // net1.transformer.n_heads
    dk_new = net2.transformer.d_model // net2.transformer.n_heads

    new_state = net2.state_dict()
    updated_state = deepcopy(new_state)
    for k in new_state:
        if k == "position_encoding" or k == "self_attn_mask":
            continue
        elif "self_attn_norm" in k.split(".") or "ffn_norm" in k.split("."):
            continue
        elif "attn_heads" in k.split("."):
            updated_state[k] = th.zeros_like(new_state[k])
            weight_name = k.split(".")
            layer_idx = int(weight_name[3])
            if layer_idx < n_layers_old:
                head_idx = int(weight_name[6])
                lst = [
                    (i // dk_old, i % dk_old)
                    for i in (head_idx * dk_new, head_idx * dk_new + dk_new)
                ]
                w = []
                if lst[0][0] == lst[1][0]:
                    if not lst[0][0] < n_head_old:
                        continue
                    weight_name_old = weight_name.copy()
                    weight_name_old[6] = str(lst[0][0])
                    k_old = ".".join(weight_name_old)
                    w.append(old_state[k_old][lst[0][1] : lst[1][1], :])
                else:
                    for prev_head_idx in range(lst[0][0], lst[1][0] + 1):
                        if not prev_head_idx < n_head_old:
                            continue
                        weight_name_old = weight_name.copy()
                        weight_name_old[6] = str(prev_head_idx)
                        k_old = ".".join(weight_name_old)

                        if prev_head_idx == lst[0][0]:
                            w_dash = old_state[k_old][lst[0][1] :, :]
                            # print(rng,w_dash.shape)
                            w.append(w_dash)

                        elif prev_head_idx == lst[1][0]:
                            w_dash = old_state[k_old][: lst[1][1], :]
                            # print(rng, w_dash.shape)
                            w.append(w_dash)
                        else:
                            w.append(old_state[k_old])
                    if w:
                        final_old_w = th.cat(w)
                        dice = [slice(dim) for dim in final_old_w.shape]
                        updated_state[k][dice] = final_old_w
        else:
            updated_state[k] = th.zeros_like(new_state[k])
            if k in old_state:
                dice = [slice(dim) for dim in old_state[k].shape]
                updated_state[k][dice] = old_state[k]
    net2.load_state_dict(updated_state)
