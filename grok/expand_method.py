import torch
import pytorch_lightning
import functools
import transformer
from training import TrainableTransformer
import numpy as np


def expand_model(
    self,
    add_dmodel: int,
    exp_method: str = "random",
) -> None:
    """Expand Transformer dmodel to dmodel + add_dmodel.

    Args:
        parent_net:(grok.transformer.Transformer) The parent model to expand from.
        add_dmodel:(int) increase in the size of d_model.
        exp_method:(str) [duplicate | random | zero] Method used to initialize new parameter.
    """
    new_d_model = self.transformer.d_model + add_dmodel
    print(f"\nExpanding to size {new_d_model}")
    self.log("d_model", new_d_model)
    teacher_model = self.transformer
    device = next(teacher_model.parameters()).device

    student_model = type(teacher_model)(
        n_layers=teacher_model.n_layers,
        n_heads=teacher_model.n_heads,
        d_model=new_d_model,
    )
    assert exp_method in [
        "duplicate",
        "random",
        "zero",
    ], "Invalid expansion method."
    with torch.no_grad():
        for (k1, _), (k2, _) in zip(
            self.transformer.named_parameters(), student_model.named_parameters()
        ):
            assert k1 == k2

            keys = k1.split(".")
            param_old_parent = functools.reduce(getattr, [self.transformer, *keys[:-1]])
            param_new_parent = functools.reduce(getattr, [student_model, *keys[:-1]])
            if type(param_old_parent) == transformer.LayerNorm:
                setattr(param_old_parent, "normalized_shape", (new_d_model,))
            param_name = keys[-1]
            param_old = getattr(param_old_parent, param_name)
            param_new = getattr(param_new_parent, param_name)

            if param_new.shape == param_old.shape:
                setattr(
                    param_old_parent,
                    param_name,
                    torch.nn.parameter.Parameter(param_old.clone()),
                )
            else:
                new_shape = param_new.shape
                old_shape = param_old.shape
                w_ = param_old.clone()
                for dim in range(len(new_shape)):
                    # m is the size  to concat in dimension `dim``
                    m = new_shape[dim] - old_shape[dim]
                    if exp_method == "duplicate":
                        idx = torch.tensor(
                            np.random.choice(range(w_.shape[dim]), size=m, replace=True)
                        ).to(device)
                        v_ = torch.index_select(w_, dim, idx)
                        w_ = torch.cat((w_, v_), dim=dim)

                    elif exp_method == "random":
                        shape_of_exta = w_.shape[:dim] + (m,) + w_.shape[dim + 1 :]
                        v_ = torch.randn(shape_of_exta).to(device)
                        w_ = torch.cat((w_, v_), dim=dim)

                    elif exp_method == "zero":
                        m = new_shape[dim] - old_shape[dim]
                        shape_of_exta = w_.shape[:dim] + (m,) + w_.shape[dim + 1 :]
                        v_ = torch.zeros(shape_of_exta).to(device)
                        w_ = torch.cat((w_, v_), dim=dim)
                setattr(
                    param_old_parent,
                    param_name,
                    torch.nn.parameter.Parameter(w_),
                )
        b = getattr(self.transformer, "position_encoding")
        setattr(
            self.transformer,
            "position_encoding",
            torch.tensor(
                self.transformer._position_encoding(
                    self.transformer.max_context_len, new_d_model
                ),
                dtype=b.dtype,
                device=b.device,
            ),
        )
    print(self.transformer)


class ExpandModelCallback(pytorch_lightning.callbacks.Callback):
    def __init__(self, expand_size: int, expand_count: float, expand_method: str):
        super().__init__()
        self.expand_size = expand_size
        self.expand_count = expand_count
        self.expand_method = expand_method
        self.N = None
        self.current_epoch = 0

    def on_epoch_end(
        self, trainer: pytorch_lightning.Trainer, pl_module: TrainableTransformer
    ):
        assert (
            trainer.max_steps or trainer.max_epochs
        ), "Please provide either max_step or max_epochs"

        current_epoch = pl_module.current_epoch

        if self.current_epoch == current_epoch:
            return
        else:
            self.current_epoch = current_epoch

        if current_epoch == 1 and not self.N:
            if trainer.max_steps:
                total_steps = min(1e5, trainer.max_steps)
                total_epochs = total_steps // pl_module.batches_per_epoch
                self.N = int(total_epochs // (self.expand_count) + 1)
                print(f"\nExpanding freq set to {self.N} epochs")
            else:
                self.N = int(trainer.max_epochs // (self.expand_count) + 1)
                print(f"\nExpanding freq set to {self.N} epochs")

        self.log("d_model", pl_module.transformer.d_model)
        if current_epoch > 0 and current_epoch % self.N == 0:
            pl_module.expand_model(self.expand_size, exp_method=self.expand_method)
