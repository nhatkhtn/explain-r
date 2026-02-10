import logging
from pathlib import Path
from functools import partial

import torch
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm, trange

from exrep.loss import SimilarityLoss
from exrep.model import init_target
from exrep.model.distill import DistilledRepresentationModel
from exrep.model.surrogate import SurrogateRepresentation
from exrep.utils import pythonize
from exrep.evaluate import dict_avg, get_classification_evaluator
from exrep.utils.data import DictOutputDatasetInterface

logger = logging.getLogger(__name__)

def validation_loop_contrastive(model: DistilledRepresentationModel, val_dataloader: DataLoader, loss_fn: SimilarityLoss, device):
    model.eval()
    val_losses = []
    for batch in val_dataloader:
        # move to device
        images = batch['inputs'].to(device)
        texts = batch['captions'].to(device)
        encodings = batch["encodings"].to(device=device, dtype=torch.float32)
        indices = batch["indices"]  # indices stay on CPU
        
        # forward pass
        with torch.inference_mode():
            queries_student, keys_student, queries_teacher, keys_teacher = model(images, texts, encodings)
            loss_dict = loss_fn(queries_student, keys_student, queries_teacher, keys_teacher, indices)

        val_losses.append(pythonize(loss_dict))

    return { f"val_{k}": v for k, v in dict_avg(val_losses).items() }

def init_objective_evaluator(name: str, **kwargs):
    """Initialize the objective evaluator based on the name. 

    Returns a loop that either computes the validation accuracy or the validation loss.
    """
    if name == 'accuracy':
        return get_classification_evaluator(kwargs['dataset'])
    
    return partial(validation_loop_contrastive, loss_fn=SimilarityLoss.get_loss(name, **kwargs))

def train_bimodal(
    model_config: dict,
    loss_config: dict,
    optimizer_config: dict,
    target_config: dict,
    training_config: dict,
    train_dataset: DictOutputDatasetInterface,
    val_dataset: DictOutputDatasetInterface,
    output_path: str | Path,
    wandb_run=None,
    log_every_n_steps: int = 10,
    device: str = "cuda",
):
    """Train local representation with student-teacher similarity matching.

    Args:
        model_config (dict): model configuration.
        loss_config (dict): loss configuration.
        optimizer_config (dict): optimizer configuration.
        target_config (dict): target model configuration.
        training_config (dict): training configuration.
        num_epochs (int, optional): number of epochs. Defaults to 10.
        log_every_n_steps (int, optional): log every n steps. Defaults to 10.
        device (str, optional): device. Defaults to "cuda".
    """
    # init the model to be explained
    target_model = init_target(**target_config, device=device)
    logger.info("Loaded target model %s", target_config["name"])

    # init dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        pin_memory=True,
        pin_memory_device=device,
        **training_config["train_dataloader"],
    )
    val_dataloader = DataLoader(
        val_dataset,
        pin_memory=True,
        pin_memory_device=device,
        **training_config["val_dataloader"],
    )

    # infer model configuration based on input
    sample_datum = train_dataset[0]
    model_config["local_dim"] = sample_datum["encodings"].shape[0]
    with torch.inference_mode():
        sample_embedding = target_model.embed_query(sample_datum['inputs'].unsqueeze(0).to(device))
        model_config["repr_dim"] = sample_embedding.shape[1]
    model_config["temperature"] = loss_config["train"]["temp_student"]
    logger.info("Model config: %s", model_config)

    # infer loss configuration based on input
    train_loss_config, validation_loss_config = loss_config['train'], loss_config['val']
    train_loss_config["data_size"] = len(train_dataset)
    validation_loss_config["data_size"] = len(val_dataset)

    # create the model
    surrogate = SurrogateRepresentation.initialize(**model_config, device=device)
    model = DistilledRepresentationModel(target_model, surrogate)

    # create the optimizer
    # note the LR is scaled according to the batch size
    optimizer_config['lr'] *= loss_config["train"]["temp_student"] * training_config["train_dataloader"]['batch_size'] / 256
    optimizer_name = optimizer_config.pop("name", "AdamW")
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), **optimizer_config)
    logger.info("Actual lr is %.6f", optimizer_config['lr'])

    # create the loss function
    train_loss_fn = SimilarityLoss.get_loss(**train_loss_config)
    validation_loop = init_objective_evaluator(**validation_loss_config)

    # checkpointing
    best_val_loss = 1.0
    best_val_acc = 0.0
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # training configs
    num_epochs = training_config["epochs"]
    val_freq = training_config.get("val_freq", 1)

    # main training loop
    logs = {"train": [], "val": []}
    steps = 0
    for epoch in trange(num_epochs):
        model.train()
        model.target_model.eval()
        
        pbar = tqdm(train_dataloader)
        for batch in pbar:
            # move to device
            images = batch['inputs'].to(device)
            texts = batch['captions'].to(device)
            encodings = batch["encodings"].to(device=device, dtype=torch.float32)
            indices = batch["indices"]  # indices stay on CPU

            # forward pass
            queries_student, keys_student, queries_teacher, keys_teacher = model(images, texts, encodings)

            # compute losses
            loss_dict = train_loss_fn(queries_student, keys_student, queries_teacher, keys_teacher, indices)
            total_loss = loss_dict["loss"]

            # update model
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # logging
            steps += 1
            if log_every_n_steps > 0 and steps % log_every_n_steps == 0:
                pbar.set_postfix_str(f"Loss: {total_loss.item():.5f}, Acc: {loss_dict.get('accuracy', 0.0):.2f}")
                # tqdm.write(f"Epoch {epoch}, Step {steps}, Loss: {total_loss.item():.5f}, Acc: {loss_dict.get('accuracy', 0.0):.2f}")
                # with logging_redirect_tqdm([logger]):
                #     logger.info("Epoch %2d, Step %4d, Loss: %.5f, Acc: %.2f", epoch, steps, total_loss.item(), loss_dict.get("accuracy", 0.0))
            logs["train"].append({"epoch": epoch} | pythonize(loss_dict))
            wandb_run.log(logs["train"][-1])

        # validation loop
        if epoch % val_freq == (val_freq - 1):
            val_loss_dict = validation_loop(model=model, val_dataloader=val_dataloader, device=device)
            logs["val"].append({"epoch": epoch} | val_loss_dict)

            # checkpoint
            if 'val_loss' in val_loss_dict:
                mean_val_loss = val_loss_dict['val_loss']
                if mean_val_loss < best_val_loss:
                    logger.info("Best val loss improved from %.6f to %.6f", best_val_loss, mean_val_loss)
                    best_val_loss = mean_val_loss
                    torch.save(model.surrogate_model, output_path / f"{wandb_run.name}_best_val-loss.pt")
                    wandb_run.save(output_path / f"{wandb_run.name}_best_val-loss.pt")

            if 'val_accuracy' in val_loss_dict:
                mean_val_acc = val_loss_dict['val_accuracy']    
                if mean_val_acc > best_val_acc:
                    logger.info("Best val acc improved from %.4f to %.4f", best_val_acc, mean_val_acc)
                    best_val_acc = mean_val_acc
                    torch.save(model.surrogate_model, output_path / f"{wandb_run.name}_best_val-acc.pt")
                    wandb_run.save(output_path / f"{wandb_run.name}_best_val-acc.pt")
                wandb_run.log({"val_accuracy": mean_val_acc})

            wandb_run.log({
                "best_val_loss": best_val_loss,
                "best_val_acc": best_val_acc,
            })

    return model, logs