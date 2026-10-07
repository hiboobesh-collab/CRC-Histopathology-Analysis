from pathlib import Path
import torch


class CheckpointManager:

    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)

        self.checkpoint_dir = self.model_dir / "checkpoints"
        self.best_dir = self.model_dir / "best"
        self.final_dir = self.model_dir / "final"

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_dir.mkdir(parents=True, exist_ok=True)
        self.final_dir.mkdir(parents=True, exist_ok=True)


    def save_epoch(
        self,
        epoch,
        model,
        optimizer,
        metrics,
    ):
        """
        Save a checkpoint after every epoch.
        """

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            **metrics,
        }

        epoch_path = (
            self.checkpoint_dir /
            f"epoch_{epoch:02d}.pth"
        )

        torch.save(checkpoint, epoch_path)

        # Always maintain the latest checkpoint.
        latest_path = self.checkpoint_dir / "latest.pth"

        torch.save(checkpoint, latest_path)

        return epoch_path


    def save_best(
        self,
        epoch,
        model,
        optimizer,
        metrics,
    ):
        """
        Save the best-performing model.
        """

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            **metrics,
        }

        best_path = self.best_dir / "resnet18_best.pth"

        torch.save(checkpoint, best_path)

        return best_path


    def save_final(
        self,
        epoch,
        model,
        optimizer,
        metrics,
    ):
        """
        Save the final model from the completed run.
        """

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            **metrics,
        }

        final_path = self.final_dir / "resnet18_final.pth"

        torch.save(checkpoint, final_path)

        return final_path
