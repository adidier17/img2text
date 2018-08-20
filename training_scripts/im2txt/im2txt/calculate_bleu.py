"""Evaluate the model using BLEU score.

This script should be run concurrently with training so that summaries show up
in TensorBoard.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os.path
import time
import json

import numpy as np
import tensorflow as tf

from im2txt import configuration
from im2txt import show_and_tell_model
# import configuration
# import show_and_tell_model
from im2txt.inference_utils import caption_generator
from im2txt.inference_utils import vocabulary
from bleu_scorer import BleuScorer

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("test_images_dir", "",
                       "Directory containing test images")
tf.flags.DEFINE_string("checkpoint_dir", "",
                       "Directory containing model checkpoints.")
tf.flags.DEFINE_string("vocab_file", "", "Text file containing the vocabulary.")
tf.flags.DEFINE_string("eval_dir", "", "Directory to write event logs.")
tf.flags.DEFINE_string("test_captions_file", "", "json file containing captions and file names for the test images")
tf.flags.DEFINE_integer("eval_interval_secs", 180,
                        "Interval between evaluation runs.")
tf.flags.DEFINE_integer("num_eval_examples", 10132,
                        "Number of examples for evaluation.")

tf.flags.DEFINE_integer("min_global_step", 2,
                        "Minimum global step to run evaluation.")

tf.logging.set_verbosity(tf.logging.INFO)

def caption_image(image, model, sess, vocab):
    # Prepare the caption generator. Here we are implicitly using the default
    # beam search parameters. See caption_generator.py for a description of the
    # available beam search parameters.
    generator = caption_generator.CaptionGenerator(model, vocab)
    captions = generator.beam_search(sess, image)
    best_caption = captions[0]
    caption = [vocab.id_to_word(w) for w in best_caption.sentence[1:-1]]
    return " ".join(caption)

def evaluate_model(sess, model, global_step, summary_writer, summary_op, vocab):
    """Computes BLEU score on the test images then computes an average bleu.

    Summaries and perplexity-per-word are written out to the eval directory.

    Args:
    sess: Session object.
    model: Instance of ShowAndTellModel; the model to evaluate.
    global_step: Integer; global step of the model checkpoint.
    summary_writer: Instance of FileWriter.
    summary_op: Op for generating model summaries.
    """
    # Log model summaries on a single batch.
    summary_str = sess.run(summary_op)
    summary_writer.add_summary(summary_str, global_step)

    start_time = time.time()
    sum_bleu = 0.
    count = 0.

    with open(FLAGS.test_captions_file, "r") as f:
        caption_dict = json.load(f)["file_caps"]

    for fname, cap in caption_dict.items():
        image_path = os.path.join(FLAGS.test_images_dir, fname)
        with tf.gfile.GFile(image_path, "rb") as f:
            image = f.read()
        test_caption = caption_image(image, model, sess, vocab)
        scorer = BleuScorer(test_caption, [cap])
        bleu_score, bleu_list = scorer.compute_score()
        sum_bleu += bleu_score
        count += 1
    bleu_avg = sum_bleu/count



    tf.logging.info("Computed bleu score at global step %d.", global_step)
    eval_time = time.time() - start_time

    tf.logging.info("BLEU = %f (%.2g sec)", bleu_avg, eval_time)

    # Log perplexity to the FileWriter.
    summary = tf.Summary()
    value = summary.value.add()
    value.simple_value = bleu_avg
    value.tag = "Bleu Score"
    summary_writer.add_summary(summary, global_step)

    # Write the Events file to the eval directory.
    summary_writer.flush()
    tf.logging.info("Finished processing evaluation at global step %d.",
                  global_step)


def run_once(model, saver, summary_writer, summary_op, vocab):
    """Evaluates the latest model checkpoint.

    Args:
    model: Instance of ShowAndTellModel; the model to evaluate.
    saver: Instance of tf.train.Saver for restoring model Variables.
    summary_writer: Instance of FileWriter.
    summary_op: Op for generating model summaries.
    """
    model_path = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    if not model_path:
        tf.logging.info("Skipping evaluation. No checkpoint found in: %s",
                    FLAGS.checkpoint_dir)
        return

    with tf.Session() as sess:
        # Load model from checkpoint.
        tf.logging.info("Loading model from checkpoint: %s", model_path)
        saver.restore(sess, model_path)
        global_step = tf.train.global_step(sess, model.global_step.name)
        tf.logging.info("Successfully loaded %s at global step = %d.",
                        os.path.basename(model_path), global_step)
        if global_step < FLAGS.min_global_step:
          tf.logging.info("Skipping evaluation. Global step = %d < %d", global_step,
                          FLAGS.min_global_step)
          return

        # Start the queue runners.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # Run evaluation on the latest checkpoint.
        try:
          evaluate_model(
              sess=sess,
              model=model,
              global_step=global_step,
              summary_writer=summary_writer,
              summary_op=summary_op,
              vocab=vocab)
        except Exception as e:  # pylint: disable=broad-except
          tf.logging.error("Evaluation failed.")
          coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


def run():
  """Runs evaluation in a loop, and logs summaries to TensorBoard."""
  # Create the evaluation directory if it doesn't exist.
  eval_dir = FLAGS.eval_dir
  if not tf.gfile.IsDirectory(eval_dir):
    tf.logging.info("Creating eval directory: %s", eval_dir)
    tf.gfile.MakeDirs(eval_dir)

  g = tf.Graph()
  with g.as_default():
    # Build the model for evaluation.
    model_config = configuration.ModelConfig()
    model = show_and_tell_model.ShowAndTellModel(model_config, mode="inference")
    model.build()

    # Create the Saver to restore model Variables.
    saver = tf.train.Saver()

    # Create the summary operation and the summary writer.
    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(eval_dir)

    g.finalize()

    vocab = vocabulary.Vocabulary(FLAGS.vocab_file)
    #Run a new evaluation run every eval_interval_secs.
    while True:
      start = time.time()
      tf.logging.info("Starting evaluation at " + time.strftime(
          "%Y-%m-%d-%H:%M:%S", time.localtime()))
      run_once(model, saver, summary_writer, summary_op, vocab)
      time_to_next_eval = start + FLAGS.eval_interval_secs - time.time()
      if time_to_next_eval > 0:
        time.sleep(time_to_next_eval)


def main(unused_argv):
    assert FLAGS.test_images_dir, "--test_images_dir is required"
    assert FLAGS.checkpoint_dir, "--checkpoint_dir is required"
    assert FLAGS.eval_dir, "--eval_dir is required"
    run()


if __name__ == "__main__":
    tf.app.run()
