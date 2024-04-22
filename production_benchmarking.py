from pathlib import Path
from time import perf_counter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, models, transforms
from torchvision.datasets import ImageNet
import evaluate
import os
from tqdm.auto import tqdm
# TODO - import device? 
# TODO - import model


class PerformanceBenchmark:
    def __init__(self, pipeline, dataset, optim_type="transfer_learning"):
        self.pipeline = pipeline
        self.dataset = dataset # TODO - dataset or dataloader?
        self.optim_type = optim_type
    
    def compute_accuracy(self):
        """
        Compute accuracy of the pipeline on the test set. 
        Collects all the predictions and labels in lists before returning the accuracy on the dataset.
        """
        metric = evaluate.load("accuracy")

        for data in data_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
        with torch.no_grad():
            outputs = model(inputs)
        _, predictions = torch.max(outputs, 1)
        metric.add_batch(predictions=predictions, references=labels)
        eval_results = metric.compute()
        
        print(f"Accuracy on test set - {eval_results:.3f}")
        return eval_results

    def compute_size(self):
        state_dict = self.pipeline.model.state_dict()
        tmp_path = Path("model.pt")
        torch.save(state_dict, tmp_path)
        # Calculate size in megabytes
        size_mb = Path(tmp_path).stat().st_size / (1024 * 1024)
        # Delete temporary file
        tmp_path.unlink()
        print(f"Model size (MB) - {size_mb:.2f}")
        return {"size_mb": size_mb}

    def time_pipeline(self, query):
        # TODO - query should be an input image?
        latencies = []
        # # Warmup
        # for _ in range(10):
        #     _ = self.pipeline(query)
        # Timed run
        # TODO - use torch.cuda.synchronize() to measure time accurately? 
        for _ in range(100):
            start_time = perf_counter()
            _ = self.pipeline(query)
            latency = perf_counter() - start_time
            latencies.append(latency)
        # Compute run statistics
        time_avg_ms = 1000 * np.mean(latencies)
        time_std_ms = 1000 * np.std(latencies)
        print(f"Average latency (ms) - {time_avg_ms:.2f} +\- {time_std_ms:.2f}")
        return {"time_avg_ms": time_avg_ms, "time_std_ms": time_std_ms}

    def run_benchmark(self):
        metrics = {}
        metrics[self.optim_type] = self.compute_size()
        metrics[self.optim_type].update(self.compute_accuracy())
        metrics[self.optim_type].update(self.time_pipeline())
        return metrics
