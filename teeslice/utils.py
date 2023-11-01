import torch
import os, sys



def load_checkpoint(
    model, model_dir=None, model_fn=None, tensors=None, optimizer=None, outps=None, ignore_tensors=None
):
    assert (model_dir is not None and model_fn is None) or (model_dir is None and model_fn is not None)
    fn = model_fn if model_fn is not None else os.path.join(model_dir, 'checkpoint.pth.tar')
    checkpoint = torch.load(fn)

    state = model.state_dict()

    n_loaded, n_ignored = 0, 0
    loaded_tensors = []
    for k_ckp in checkpoint['state_dict']:
        if ( (tensors is not None and k_ckp not in tensors) or
             (ignore_tensors is not None and k_ckp in ignore_tensors)
        ):
            n_ignored += 1
            continue

        k_st = tensors[k_ckp] if tensors is not None else k_ckp
        if k_st in state:
            state[k_st] = checkpoint['state_dict'][k_ckp].clone()
            n_loaded += 1
            loaded_tensors.append(k_st)
        else:
            n_ignored += 1
            
    print('Loading checkpoint: {}\n - Tensors loaded {}\n - Tensors ignored {}.\n'.format(fn, n_loaded, n_ignored))
    model.load_state_dict(state)

    if 'keep_flags' in checkpoint:
        model.load_keep_flags(checkpoint['keep_flags'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])

    if outps is None:
        return loaded_tensors
    else:
        return loaded_tensors, tuple([checkpoint[k] for k in outps])

def save_checkpoint(model_dir, state, ignore_tensors=None):
    checkpoint_fn = os.path.join(model_dir, 'checkpoint.pth.tar')
    if ignore_tensors is not None:
        for p in ignore_tensors.values():
            if p in state['state_dict']:
                del state['state_dict'][p]
    torch.save(state, checkpoint_fn)
