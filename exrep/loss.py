import logging
from typing import Literal

import torch

class SimilarityLoss(torch.nn.Module):
    """Base class for teacher-student similarity losses."""
    def __init__(self, temp_student: float = 1.0, temp_teacher: float = 1.0):
        super().__init__()
        self.temp_student = temp_student
        self.temp_teacher = temp_teacher

    @staticmethod
    def get_loss(name: str, **kwargs):
        """Instantiate a loss function based on its name."""
        if name == "KDLoss":
            return KDLoss(**kwargs)
        elif name == "KDNaiveLoss":
            return KDNaiveLoss(**kwargs)
        elif name == 'SymmetricKDLoss':
            return SymmetricKDLoss(**kwargs)
        elif name == 'SymmetricKDNaiveLoss':
            return SymmetricKDNaiveLoss(**kwargs)
        else:
            raise ValueError(f"Unknown loss name: {name}")

    @staticmethod
    def get_scaled_sim(queries: torch.Tensor, keys: torch.Tensor, temp: float):
        """Compute cosine similarity with temperature scaling."""
        q = torch.nn.functional.normalize(queries, dim=-1)
        k = torch.nn.functional.normalize(keys, dim=-1)
        return q @ k.T / temp

    def get_teacher_sim(self, queries: torch.Tensor, keys: torch.Tensor):
        """Convinience function to compute similarity with teacher temperature."""
        return self.get_scaled_sim(queries, keys, self.temp_teacher)

    def get_student_sim(self, queries: torch.Tensor, keys: torch.Tensor):
        """Convinience function to compute similarity with student temperature."""
        return self.get_scaled_sim(queries, keys, self.temp_student)

class KDNaiveLoss(SimilarityLoss):
    """This loss use the same interface as KDLoss, but does not use the moving average mechanism."""
    def __init__(self,
                 data_size: int = None,
                 gamma1: float = 1.0,
                 gamma2: float = 1.0,
                 temp_student: float = 1.0,
                 temp_teacher: float = 1.0,
                 weights: torch.Tensor | None = None,
                 variant: Literal['kl', 'ce'] = 'kl',
                 ):
        """Knowledge distillation loss

        Args:
            data_size (int): size of the dataset, used to determine the length of u vector
            gamma1 (float): moving average coefficient for u update
            gamma2 (float): moving average coefficient for v update
            temp_student (float, optional): temperature for student. Defaults to 1.0.
            temp_teacher (float, optional): temperature for teacher. Defaults to 1.0.
            weights (torch.Tensor, optional): weights for each sample. Defaults to None.
        """
        super().__init__(temp_student=temp_student, temp_teacher=temp_teacher)

        logging.info("Initializing KDNaiveLoss with student temp %.2f and teacher temp %.2f", temp_student, temp_teacher)

        self.weights = weights
        self.variant = variant

        assert variant in ('kl', 'ce'), f"Unknown variant {variant}"


    def forward(self,
                queries_student: torch.Tensor,
                keys_student: torch.Tensor,
                queries_teacher: torch.Tensor,
                keys_teacher: torch.Tensor,
                index: torch.Tensor,
                ):
        sim_student = self.get_student_sim(queries_student, keys_student)
        sim_teacher = self.get_teacher_sim(queries_teacher, keys_teacher)

        if self.variant == 'kl':
            logprob_student = torch.log_softmax(sim_student, dim=-1)
            logprob_teacher = torch.log_softmax(sim_teacher, dim=-1)

            loss_batch = torch.nn.functional.kl_div(
                input=logprob_student,
                target=logprob_teacher,
                reduction="batchmean",
                log_target=True,
            )

            with torch.inference_mode():
                accuracy = torch.mean((sim_student.argmax(dim=-1) == sim_teacher.argmax(dim=-1)).float())
            return {"loss": loss_batch, "accuracy": accuracy}
        else:
            loss_batch = torch.nn.functional.cross_entropy(
                input=sim_student,
                target=torch.softmax(sim_teacher, dim=-1),
            )
            return {"loss": loss_batch}

class KDLoss(SimilarityLoss):
    def __init__(self,
                 data_size: int,
                 gamma1: float = 1.0,
                 gamma2: float = 1.0,
                 temp_student: float = 1.0,
                 temp_teacher: float = 1.0,
                 weights: torch.Tensor | None = None,
                 ):
        """Knowledge distillation loss

        Args:
            data_size (int): size of the dataset, used to determine the length of u vector
            gamma1 (float): moving average coefficient for u update
            gamma2 (float): moving average coefficient for v update
            temp_student (float, optional): temperature for student. Defaults to 1.0.
            temp_teacher (float, optional): temperature for teacher. Defaults to 1.0.
            weights (torch.Tensor, optional): weights for each sample. Defaults to None.
        """
        super().__init__(temp_student=temp_student, temp_teacher=temp_teacher)
        self.data_size = data_size
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.weights = weights

        self.u = torch.zeros(data_size, device="cpu").reshape(-1, 1)
        self.v = torch.zeros(data_size, device="cpu").reshape(-1, 1)

    def forward(self,
                queries_student: torch.Tensor,
                keys_student: torch.Tensor,
                queries_teacher: torch.Tensor,
                keys_teacher: torch.Tensor,
                index: torch.Tensor,
                ):
        # update u
        # logits are exp(cosine_similarity / temperature)
        sim_teacher = self.get_teacher_sim(queries_teacher, keys_teacher)
        logits_teacher = torch.exp(sim_teacher)                                        # shape: (B^x, B^k)
        with torch.no_grad():
            u = self.u[index].to(queries_teacher.device)                               # shape: (B^x, 1)
            if u.sum() == 0:
                gamma1 = 1.0
            else:
                gamma1 = self.gamma1
            u = (1 - gamma1) * u + gamma1 * torch.mean(logits_teacher, dim=-1, keepdim=True)
            self.u[index] = u.cpu()

        # update v
        sim_student = self.get_student_sim(queries_student, keys_student)
        logits_student = torch.exp(sim_student)                                        # shape: (B^x, B^k)
        with torch.no_grad():
            v = self.v[index].to(queries_student.device)                               # shape: (B^x, 1)
            if v.sum() == 0:
                gamma2 = 1.0
            else:
                gamma2 = self.gamma2
            v = (1 - gamma2) * v + gamma2 * torch.mean(logits_student, dim=-1, keepdim=True)
            self.v[index] = v.cpu()
        g_student_batch = torch.mean(logits_student, dim=-1, keepdim=True) / logits_student # shape: (B^x, B^k)

        # get weights
        if self.weights is not None:
            weights = self.weights[index].to(queries_student.device)
        else:
            weights = torch.ones_like(index, dtype=torch.float32, device=queries_student.device)
        weights = weights.reshape(-1, 1)    # add a last dimension

        # compute gradient estimator
        grad_estimator = torch.mean(logits_teacher.detach() / u * logits_student.detach() / v * g_student_batch * weights)

        with torch.inference_mode():
            accuracy = torch.mean((sim_student.argmax(dim=-1) == sim_teacher.argmax(dim=-1)).float())
            logprob_student = torch.log_softmax(sim_student, dim=-1)
            logprob_teacher = torch.log_softmax(sim_teacher, dim=-1)

            kl_loss = torch.nn.functional.kl_div(
                input=logprob_student,
                target=logprob_teacher,
                reduction="batchmean",
                log_target=True,
            )

        return {"loss": grad_estimator, "accuracy": accuracy, "kl_loss": kl_loss}

class SymmetricKDLoss(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.i2t_loss = KDLoss(**kwargs)
        self.t2i_loss = KDLoss(**kwargs)

    def forward(self, queries_student, keys_student, queries_teacher, keys_teacher, index):
        # query: image, key: text
        i2t_loss_dict = self.i2t_loss(queries_student, keys_student, queries_teacher, keys_teacher, index)
        t2i_loss_dict = self.t2i_loss(keys_student, queries_student, keys_teacher, queries_teacher, index)

        # take the average over all keys in the loss dict
        loss_dict = {
            "loss": (i2t_loss_dict["loss"] + t2i_loss_dict["loss"]) / 2,
            "accuracy": (i2t_loss_dict["accuracy"] + t2i_loss_dict["accuracy"]) / 2,
            "kl_loss": (i2t_loss_dict["kl_loss"] + t2i_loss_dict["kl_loss"]) / 2,
            "i2t_accuracy": i2t_loss_dict["accuracy"],
            "i2t_kl_loss": i2t_loss_dict["kl_loss"],
        }
        return loss_dict

class SymmetricKDNaiveLoss(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.i2t_loss = KDNaiveLoss(**kwargs)
        self.t2i_loss = KDNaiveLoss(**kwargs)

    def forward(self, queries_student, keys_student, queries_teacher, keys_teacher, index):
        # query: image, key: text
        i2t_loss_dict = self.i2t_loss(queries_student, keys_student, queries_teacher, keys_teacher, index)
        t2i_loss_dict = self.t2i_loss(keys_student, queries_student, keys_teacher, queries_teacher, index)

        # take the average over all keys in the loss dict
        loss_dict = {
            "loss": (i2t_loss_dict["loss"] + t2i_loss_dict["loss"]) / 2,
            "accuracy": (i2t_loss_dict["accuracy"] + t2i_loss_dict["accuracy"]) / 2,
            "i2t_accuracy": i2t_loss_dict["accuracy"],
            "i2t_loss": i2t_loss_dict["loss"],
        }
        return loss_dict