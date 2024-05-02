import time
import os
import json
import matplotlib.pyplot as plt


class Reporter:
    """Write the training performance and evaluating result to the report file."""
    def __init__(self, note: str="", report_folder: str="report") -> None:
        """Initialize the reporter.

        :param str note: the note of the experiment
        """
        self.note = note
        self.report_folder = report_folder
        os.makedirs(self.report_folder, exist_ok=True)
        self.report_files = {'training_report': os.path.join(self.report_folder, 'training_report.txt')}
        self.write_log(f"note: {self.note}")

        self.training_record = {'epoch': [], 'train_loss': [], 'train_accuracy': [], 'eval_loss': [], 'eval_accuracy': []}


    def update_training_record(self, epoch: int, train_loss: float, train_accuracy: float, eval_loss: float, eval_accuracy: float):
        """Update the training record.
        Args:
            epoch (int): the epoch number
            train_loss (float): the training loss
            train_accuracy (float): the training accuracy
            eval_loss (float): the evaluating loss
            eval_accuracy (float): the evaluating accuracy
        """
        self.training_record['epoch'].append(epoch)
        self.training_record['train_loss'].append(train_loss)
        self.training_record['train_accuracy'].append(train_accuracy)
        self.training_record['eval_loss'].append(eval_loss)
        self.training_record['eval_accuracy'].append(eval_accuracy)


    def plot_training_performance(self):
        """Plot the training performance."""
        plt.figure()
        plt.plot(self.training_record['epoch'], self.training_record['train_loss'], label='train loss')
        plt.plot(self.training_record['epoch'], self.training_record['eval_loss'], label='eval loss')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('Training and Evaluating Loss')
        plt.savefig(os.path.join(self.report_folder, 'training_loss.png'))

        plt.figure()
        plt.plot(self.training_record['epoch'], self.training_record['train_accuracy'], label='train accuracy')
        plt.plot(self.training_record['epoch'], self.training_record['eval_accuracy'], label='eval accuracy')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.title('Training and Evaluating Accuracy')
        plt.savefig(os.path.join(self.report_folder, 'training_accuracy.png'))


    def upload_all_reports_to_blob(self):
        """Upload the report files to the blob storage.

        The files would be uploaded to the folder: report/report_{YYmmdd_HHMMSS} in the blob storage.
        The file training_report.txt is the training report, recording training loss and accuracy.
        The file inference_report.json is the inference report, including training hyperparameters and inference result.
        """
        # run = Run.get_context().parent
        # ws = run.experiment.workspace
        # def_blob_store = ws.get_default_datastore()

        # blob_path = f"report/{self.report_folder}_{time.strftime('%Y%m%d-%H%M%S', time.localtime())}"
        # files = list(self.report_files.values())
        # def_blob_store.upload_files(files=files, target_path=blob_path, overwrite=True)
        # print(f"Uploaded the report files to the blob storage: {blob_path}")
        pass


    def write_inference_report_to_json(self, json_dict: dict, report_file_prefix: str):
        """Write the inference report to a json file.

        :param: dict json_dict: the dictionary to be written to the json file
        :param: str report_file_prefix: the prefix of the report file name
        :return: None
        """
        self.report_files[f'{report_file_prefix}_inference_report'] = os.path.join(self.report_folder, f'{report_file_prefix}_inference_report.json')
        report_file_path = self.report_files[f'{report_file_prefix}_inference_report']
        json.dump(json_dict, open(report_file_path, 'w'), indent=4)


    def write_training_performance(self, epoch: int, train_loss: float, train_accuracy: float, eval_loss: float, eval_accuracy: float):
        """
        Write the training performance to the report file.
        Args:
            epoch (int): the epoch number
            train_loss (float): the training loss
            train_accuracy (float): the training accuracy
            eval_loss (float): the evaluating loss
            eval_accuracy (float): the evaluating accuracy
        """
        self.update_training_record(epoch, train_loss, train_accuracy, eval_loss, eval_accuracy)
        log = "epoch: %d\ttrain loss: %.4f\ttrain accuracy: %.4f\teval loss: %.4f\teval accuracy: %.4f" \
                % (epoch, train_loss, train_accuracy, eval_loss, eval_accuracy)
        self.write_log(log)


    def write_evaluating_result(self, rouge_score):
        """
        Write the evaluating result to the report file.
        Args:
            rouge_score (dict): the rouge score
        """
        self.write_log("evaluating result:")

        log = "rouge score:"
        for key, value in rouge_score.items():
            log += f"\n{key}: {value}"
        self.write_log(log)


    def write_log(self, log):
        """Write the log to the report file, and print it in the console.

        :param: str log: the log to be written to the report file
        :return: None
        """
        print(log)
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        log = f"{current_time} {log}"
        with open(self.report_files['training_report'], "a") as f:
            f.write(log + "\n")
