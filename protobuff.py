# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 02:38:42 2020

@author: Danis
"""

import os
import os.path as osp
import tensorflow as tf
from keras.models import load_model
from keras import backend as K
from tensorflow.python.platform import gfile



def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        # Graph -> GraphDef ProtoBuf
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph



def to_protobuff(path):
    #K.set_learning_phase(0)
    #load model
    model = load_model(path)
    #for l in model.layers: l.trainable = False
    model.summary()
    #model outputs
    output = model.outputs[0].name
    #model inputs
    _input = model.inputs[0].name
    #getting frozen graph
    frozen_graph = freeze_session(K.get_session(),
                                  output_names=[out.op.name for out in model.outputs])
    #storing protobuff file
    tf.train.write_graph(frozen_graph, 'pb_model', 'word_pred_Model4.pb', 
                         as_text=False)
    pb_path = './pb_model/word_pred_Model4.pb'
    return _input, output, pb_path
    
def load_pb_nodel(path='word_pred_Model4.h5'):
    _input, output, pb_path = to_protobuff(path)
    #creating session
    sess = tf.InteractiveSession()
    
    
    
    f = gfile.FastGFile(pb_path, 'rb')
    graph_def = tf.GraphDef()
    # Parses a serialized binary message into the current message.
    graph_def.ParseFromString(f.read())
    f.close()
    
    sess.graph.as_default()
    
    
    
    # Import a serialized TensorFlow `GraphDef` protocol buffer
    # and place into the current default `Graph`.
    tf.import_graph_def(graph_def)
    
    
    
    #getting input and output tesnors
    in_tensor = sess.graph.get_tensor_by_name(_input)
    out_tensor = sess.graph.get_tensor_by_name(output)
    return in_tensor, out_tensor, sess
