import time

import torch


class Trainer:

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        checkpoint_manager,
        patience=3,
    ):

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.criterion = criterion
        self.optimizer = optimizer

        self.device = device

        self.checkpoint_manager = checkpoint_manager

        self.patience = patience


    def train_one_epoch(self):

        self.model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in self.train_loader:

            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(images)

            loss = self.criterion(
                outputs,
                labels
            )

            loss.backward()

            self.optimizer.step()

            running_loss += (
                loss.item() *
                images.size(0)
            )

            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

        loss = running_loss / total
        accuracy = correct / total

        return loss, accuracy


    @torch.no_grad()
    def validate(self):

        self.model.eval()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in self.val_loader:

            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)

            loss = self.criterion(
                outputs,
                labels
            )

            running_loss += (
                loss.item() *
                images.size(0)
            )

            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

        loss = running_loss / total
        accuracy = correct / total

        return loss, accuracy


    def fit(
        self,
        epochs,
        history,
    ):

        best_val_loss = float("inf")
        best_val_accuracy = 0.0
        best_epoch = 0

        epochs_without_improvement = 0

        total_start = time.time()

        for epoch in range(1, epochs + 1):

            print()
            print("=" * 70)
            print(f"EPOCH {epoch}/{epochs}")
            print("=" * 70)

            epoch_start = time.time()

            train_loss, train_accuracy = (
                self.train_one_epoch()
            )

            val_loss, val_accuracy = (
                self.validate()
            )

            epoch_time = (
                time.time() - epoch_start
            ) / 60

            print(
                f"Train Loss     : {train_loss:.6f}"
            )

            print(
                f"Train Accuracy : "
                f"{train_accuracy * 100:.4f}%"
            )

            print(
                f"Val Loss       : {val_loss:.6f}"
            )

            print(
                f"Val Accuracy   : "
                f"{val_accuracy * 100:.4f}%"
            )

            print(
                f"Epoch Time     : "
                f"{epoch_time:.2f} minutes"
            )

            # ------------------------------------------------
            # History
            # ------------------------------------------------

            history["train_loss"].append(
                train_loss
            )

            history["train_accuracy"].append(
                train_accuracy
            )

            history["val_loss"].append(
                val_loss
            )

            history["val_accuracy"].append(
                val_accuracy
            )

            history["epoch_time_minutes"].append(
                epoch_time
            )

            # ------------------------------------------------
            # Checkpoint metrics
            # ------------------------------------------------

            metrics = {
                "model_name": "resnet18",
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            }

            # ------------------------------------------------
            # Save EVERY epoch
            # ------------------------------------------------

            epoch_path = (
                self.checkpoint_manager.save_epoch(
                    epoch,
                    self.model,
                    self.optimizer,
                    metrics,
                )
            )

            print(
                f"Checkpoint saved: {epoch_path}"
            )

            # ------------------------------------------------
            # Best model
            # ------------------------------------------------

            improved = False

            if val_loss < best_val_loss:

                improved = True

            elif (
                val_loss == best_val_loss
                and val_accuracy > best_val_accuracy
            ):

                improved = True

            if improved:

                best_val_loss = val_loss
                best_val_accuracy = val_accuracy
                best_epoch = epoch

                epochs_without_improvement = 0

                best_path = (
                    self.checkpoint_manager.save_best(
                        epoch,
                        self.model,
                        self.optimizer,
                        metrics,
                    )
                )

                print("? NEW BEST MODEL")
                print(
                    f"Best epoch       : {best_epoch}"
                )
                print(
                    f"Best val loss    : "
                    f"{best_val_loss:.6f}"
                )
                print(
                    f"Best val accuracy: "
                    f"{best_val_accuracy * 100:.4f}%"
                )

            else:

                epochs_without_improvement += 1

                print(
                    f"No improvement: "
                    f"{epochs_without_improvement}/"
                    f"{self.patience}"
                )

            # ------------------------------------------------
            # Save current best information
            # ------------------------------------------------

            history["best_epoch"] = best_epoch
            history["best_val_loss"] = best_val_loss
            history["best_val_accuracy"] = (
                best_val_accuracy
            )

            # ------------------------------------------------
            # Early stopping
            # ------------------------------------------------

            if (
                epochs_without_improvement
                >= self.patience
            ):

                print()
                print(
                    "Early stopping triggered."
                )

                break

        # ----------------------------------------------------
        # Final model
        # ----------------------------------------------------

        final_metrics = {
            "model_name": "resnet18",
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        }

        final_path = (
            self.checkpoint_manager.save_final(
                epoch,
                self.model,
                self.optimizer,
                final_metrics,
            )
        )

        total_time = (
            time.time() - total_start
        ) / 60

        history["completed_epochs"] = epoch

        history["total_training_time_minutes"] = (
            total_time
        )

        history["best_epoch"] = best_epoch

        history["best_val_loss"] = (
            best_val_loss
        )

        history["best_val_accuracy"] = (
            best_val_accuracy
        )

        print()
        print("=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)

        print(
            f"Completed epochs : {epoch}"
        )

        print(
            f"Best epoch       : {best_epoch}"
        )

        print(
            f"Best val loss    : "
            f"{best_val_loss:.6f}"
        )

        print(
            f"Best val accuracy: "
            f"{best_val_accuracy * 100:.4f}%"
        )

        print(
            f"Total time       : "
            f"{total_time:.2f} minutes"
        )

        print(
            f"Final model      : {final_path}"
        )

        return history
