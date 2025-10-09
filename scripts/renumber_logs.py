import os
import tensorflow as tf
import fire
import shutil


def read_tfrecord_file(file_path):
    events = []
    for record in tf.data.TFRecordDataset(file_path):
        event = tf.compat.v1.Event.FromString(record.numpy())
        events.append(event)
    return events


def renumber_steps(events, starting_step=0):
    for event in events:
        # Check if the 'step' field is in the 'Event' object's attributes
        if hasattr(event, "step"):
            event.step += starting_step
    return events, (
        events[-1].step if events and hasattr(events[-1], "step") else starting_step
    )


def write_tfrecord_file(events, file_path):
    with tf.io.TFRecordWriter(file_path) as writer:
        for event in events:
            writer.write(event.SerializeToString())


def renumber_logs_in_directory(directory_path):
    """
    Renumber steps in TensorBoard log files in a specified directory.

    Args:
    directory_path (str): Path to the directory containing log files.
    """
    # make directory _logs_renumbered if it doesn't exist
    new_directory = directory_path.replace("_logs", "_logs_renumbered")
    if os.path.exists(new_directory):
        shutil.rmtree(new_directory)
    os.makedirs(new_directory, exist_ok=True)

    log_files = [
        os.path.join(directory_path, f)
        for f in os.listdir(directory_path)
        if os.path.isfile(os.path.join(directory_path, f))
    ]
    log_files.sort()
    current_step = 0

    for log_file in log_files:
        events = read_tfrecord_file(log_file)
        events, current_step = renumber_steps(events, current_step)
        new_file_path = os.path.join(new_directory, os.path.basename(log_file))
        write_tfrecord_file(events, new_file_path)
        print(
            f"Processed {log_file} -> {new_file_path} with ending step: {current_step}"
        )


if __name__ == "__main__":
    fire.Fire(renumber_logs_in_directory)
