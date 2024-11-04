import csv, os
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from partA.global_values import CSV_FILENAME, USE_GRAD


class OptimalParameters:
    def __init__(
        self,
        convergence_at_iteration,
        learning_rate,
        momentum,
        epoch,
        mini_batch_size,
        function_name,
        mse=None,
        lmb=None,
    ):
        self.convergence_at_iteration = convergence_at_iteration
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.epoch = epoch
        self.mini_batch_size = mini_batch_size
        self.function_name = function_name
        self.mse = mse
        self.lmb = lmb

    def parameter_comparison(self, **kwargs):
        """
        This function checks if the current parameters used resulted in a faster convergence than the current fastest convergence.
        If true, the current parameters are saved.
        Also passes the arguments from this function to add_to_csv
        """
        epoch = kwargs.get("epoch", None)
        convergence = kwargs.get("convergence", None)
        lr = kwargs.get("lr", None)
        mom = kwargs.get("mom", None)
        mb_size = kwargs.get("mb_size", None)
        mse = kwargs.get("mse", None)
        lmb = kwargs.get("lmb", None)

        self.add_to_csv(
            convergence,
            lr,
            epoch,
            mom,
            mb_size,
            mse,
            lmb,
        )

        if epoch and self.epoch and epoch < self.epoch or epoch and self.epoch is None:
            if convergence and convergence < self.convergence_at_iteration:
                self.convergence_at_iteration = convergence
                self.epoch = epoch
                self.learning_rate = lr
                self.momentum = mom
                self.mini_batch_size = mb_size
                self.mse = mse
                self.lmb = lmb
            return
        elif epoch is None and convergence < self.convergence_at_iteration:
            self.convergence_at_iteration = convergence
            self.learning_rate = lr
            self.momentum = mom
            self.mse = mse
            self.lmb = lmb

    def print_optimal_parameters(self, method_name):
        print(method_name + ": ")
        if self.learning_rate:
            print("LR: " + str(self.learning_rate))
        if self.momentum:
            print("Mom: " + str(self.momentum))
        if self.lmb:
            print("Lmb: " + str(self.lmb))
        if self.mini_batch_size:
            print("Mini-batch size: " + str(self.mini_batch_size))
        print("Converged at:")
        if self.epoch:
            print("Epoch: " + str(self.epoch))
        if self.convergence_at_iteration:
            print("Iteration: " + str(self.convergence_at_iteration))
        if self.mse:
            print("Mse: " + str(self.mse))
        self.add_to_csv(
            self.convergence_at_iteration,
            self.learning_rate,
            self.epoch,
            self.momentum,
            self.mini_batch_size,
            self.mse,
            self.lmb,
            "optimal_params_result.csv",
        )

    def add_to_csv(
        self, convergence, lr, epoch, mom, mb_size, mse, lmb, file_name=None
    ):
        """
        Adds the current gradient descent execution to a csv file named in partA.py
        """
        if file_name is None:
            file_name = CSV_FILENAME
        filepath = f"partA/results/{file_name}"

        if not os.path.exists(filepath):
            header = [
                "function name",
                "learning rate",
                "momentum",
                "lambda",
                "Uses grad()",
                "mini batch size",
                "epochs",
                "iterations",
                "mse",
            ]
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(header)

        with open(filepath, mode="a", newline="") as file:
            writer = csv.writer(file)
            row = []
            row.append(self.function_name)
            row.append(lr or "N/A")
            row.append(mom or "N/A")
            row.append(lmb or "N/A")
            row.append(USE_GRAD)
            row.append(mb_size or "N/A")
            row.append(epoch or "N/A")
            row.append(convergence or "N/A")
            row.append(mse or "N/A")
            writer.writerow(row)
