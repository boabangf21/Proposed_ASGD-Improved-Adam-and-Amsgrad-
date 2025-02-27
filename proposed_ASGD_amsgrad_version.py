""" Made Use of Code from https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/python/training/adam.py"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# +
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.training import optimizer
from decimal import Decimal
import contextlib
import functools
import warnings
import math
from tensorflow.keras import mixed_precision
import mpmath
import sys
import time
import threading
import random
import numpy as np
#import ray
import tensorflow.compat.v2 as tf
from tensorflow.python.ops import math_ops

tf.compat.v1.enable_v2_behavior()
tf.config.run_functions_eagerly


# -

class Proposed_ASGD_amsgrad_version(optimizer.Optimizer):

    def __init__(self, learning_rate_min=0.0001, learning_rate_max=0.01, adaptive_variable=0.5, beta1=0.9, beta2=0.99, epsilon=1e-8,
               use_locking=False, name="Proposed_ASGD_amsgrad_version"):
        super(Proposed_ASGD_amsgrad_version, self).__init__(use_locking, name)
        self._lr_min = learning_rate_min
        self._lr_max = learning_rate_max
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon
        self._adaptive = adaptive_variable

        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t_min = None #added by boabang et al.
        self._lr_t_max =None  #added by boabang et al.
        self._beta1_t = None
        self._beta2_t = None
        self._epsilon_t = None
        self._adaptive_t = None
        # Created in SparseApply if needed.
        self._updated_lr = None
        
    def _get_beta_accumulators(self):
        with ops.init_scope():
            if context.executing_eagerly():
                graph = None
            else:
                graph = ops.get_default_graph()
            return (self._get_non_slot_variable("beta1_power", graph=graph),
              self._get_non_slot_variable("beta2_power", graph=graph))

    def _create_slots(self, var_list):
        # Create the beta1 and beta2 accumulators on the same device as the first
        # variable. Sort the var_list to make sure this device is consistent across
        # workers (these need to go on the same PS, otherwise some updates are
        # silently ignored).
        first_var = min(var_list, key=lambda x: x.name)
        self._create_non_slot_variable(initial_value=self._beta1,
                                       name="beta1_power",
                                       colocate_with=first_var)
        self._create_non_slot_variable(initial_value=self._beta2,
                                       name="beta2_power",
                                       colocate_with=first_var)

        # Create slots for the first and second moments.
        for v in var_list:
            self._zeros_slot(v, "m", self._name)
            self._zeros_slot(v, "v", self._name)
            self._zeros_slot(v, "vhat", self._name)

    def _prepare(self):
        lr_min = self._call_if_callable(self._lr_min)
        lr_max = self._call_if_callable(self._lr_max)
        beta1 = self._call_if_callable(self._beta1)
        beta2 = self._call_if_callable(self._beta2)
        epsilon = self._call_if_callable(self._epsilon)
        adaptive = self._call_if_callable(self._adaptive)
        self._lr_t_min = ops.convert_to_tensor(lr_min, name="learning_rate_min")
        self._lr_t_max = ops.convert_to_tensor(lr_max, name="learning_rate_max")
        self._beta1_t = ops.convert_to_tensor(beta1, name="beta1")
        self._beta2_t = ops.convert_to_tensor(beta2, name="beta2")
        self._epsilon_t = ops.convert_to_tensor(epsilon, name="epsilon")
        #self._adaptive_t = ops.convert_to_tensor(adaptive, name="epsilon")
    #@tf.function(jit_compile=True)
    def _apply_dense(self, grad, var):
        beta1_power, beta2_power = self._get_beta_accumulators()
        beta1_power = math_ops.cast(beta1_power, var.dtype.base_dtype)
        beta2_power = math_ops.cast(beta2_power, var.dtype.base_dtype)
        lr_t_min = math_ops.cast(self._lr_t_min, var.dtype.base_dtype)
        lr_t_max = math_ops.cast(self._lr_t_max, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)
        adaptive = math_ops.cast(self._adaptive, var.dtype.base_dtype)
        lr_min = (lr_t_min * math_ops.sqrt(1 - beta2_power) / (1 - beta1_power))  # added by boabang et al.
        lr_max = (lr_t_max * math_ops.sqrt(1 - beta2_power) / (1 - beta1_power))
        # m_t = beta1 * m + (1 - beta1) * g_t
        m = self.get_slot(var, "m")
        m_scaled_g_values = grad * (1 - beta1_t)
        m_t = state_ops.assign(m, beta1_t * m + m_scaled_g_values, use_locking=self._use_locking)  

        # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
        v = self.get_slot(var, "v")
        v_scaled_g_values = (grad * grad) * (1 - beta2_t)
        v_t = state_ops.assign(v, beta2_t * v + v_scaled_g_values, use_locking=self._use_locking)

        vhat = self.get_slot(var, "vhat")
        vhat_t = state_ops.assign(vhat, math_ops.maximum(v_t, vhat))
        #vhat_t = state_ops.assign(vhat, math_ops.maximum(v_t, vhat))
        v_p = math_ops.sqrt(vhat_t)
        g_min_mean = ((tf.math.sign((tf.compat.v1.math.reduce_mean((vhat_t))) - (vhat_t)) + 1) /2)  
        g_max_mean =  ((tf.math.sign((vhat_t) - (tf.compat.v1.math.reduce_mean((vhat_t)))) + 1)/ 2)
          
            
       # float ( tf.math.count_nonzero(g_max)) <(float(tf.math.count_nonzero(g_min_mean))) and
       # @tf.function
        var_update=tf.cond(tf.math.less( ( ( tf.math.count_nonzero(g_max_mean))),  tf.cast(adaptive, int64)*  ((tf.math.count_nonzero(g_min_mean)))), lambda:  state_ops.assign_sub(var,
                                          lr_min * m_t / (v_p + epsilon_t), use_locking=self._use_locking), lambda:    state_ops.assign_sub(var, lr_max * m_t / (v_p + epsilon_t),
                                          use_locking=self._use_locking))
        
        return control_flow_ops.group(*[var_update, m_t, v_t, vhat])

   # @tf.function(jit_compile=True)
    def _resource_apply_dense(self, grad, var):
        beta1_power, beta2_power = self._get_beta_accumulators()
        beta1_power = math_ops.cast(beta1_power, var.dtype.base_dtype)
        beta2_power = math_ops.cast(beta2_power, var.dtype.base_dtype)
        lr_t_min = math_ops.cast(self._lr_t_min, var.dtype.base_dtype)
        lr_t_max = math_ops.cast(self._lr_t_max, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)
        adaptive = math_ops.cast(self._adaptive, var.dtype.base_dtype)
        lr_min = (lr_t_min * math_ops.sqrt(1 - beta2_power) / (1 - beta1_power))
        lr_max = (lr_t_max * math_ops.sqrt(1 - beta2_power) / (1 - beta1_power))
        # m_t = beta1 * m + (1 - beta1) * g_t
        m = self.get_slot(var, "m")
        m_scaled_g_values = grad * (1 - beta1_t)
        m_t = state_ops.assign(m, beta1_t * m + m_scaled_g_values, use_locking=self._use_locking)

        # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
        v = self.get_slot(var, "v")
        v_scaled_g_values = (grad * grad) * (1 - beta2_t)
        v_t = state_ops.assign(v, beta2_t * v + v_scaled_g_values, use_locking=self._use_locking)

        vhat = self.get_slot(var, "vhat")
        vhat_t = state_ops.assign(vhat, math_ops.maximum(v_t, vhat))
        v_p = math_ops.sqrt(vhat_t)
        
        g_min_mean = ((tf.math.sign((tf.compat.v1.math.reduce_mean((vhat_t))) - (vhat_t)) + 1) /2)  
        g_max_mean =  ((tf.math.sign((vhat_t) - (tf.compat.v1.math.reduce_mean((vhat_t)))) + 1)/ 2)
          
        var_update=tf.cond(tf.math.less( ( ( tf.math.count_nonzero(g_max_mean))), tf.cast(adaptive, int64)*  ((tf.math.count_nonzero(g_min_mean)))), lambda:  state_ops.assign_sub(var,
                                          lr_min * m_t / (v_p + epsilon_t), use_locking=self._use_locking), lambda:    state_ops.assign_sub(var, lr_max * m_t / (v_p + epsilon_t),
                                          use_locking=self._use_locking))    
       # float ( tf.math.count_nonzero(g_max)) <(float(tf.math.count_nonzero(g_min_mean))) and
        #@tf.function
       # if   tf.math.less( ( ( tf.math.count_nonzero(g_max_mean))),  ((tf.math.count_nonzero(g_min_mean)))):
              #var.assign_sub((m * alpha_max * (g_max_mean))  / (tf.sqrt(v) - coefficients["epsilon"]) + (m * alpha_standard * (g_min_mean) ) / (tf.sqrt(v) - coefficients["epsilon"]))
            #     var_update = state_ops.assign_sub(var,
            #                              lr_min * m_t / (v_p + epsilon_t),
            #                              use_locking=self._use_locking)
                
        #else:
                #var.assign_sub((m * alpha_max ) / (tf.pow(v,1/8) - coefficients["epsilon"]))
         #       var_update = state_ops.assign_sub(var,
             #                             lr_max * m_t / (v_p + epsilon_t),
          #                                use_locking=self._use_locking)
        
        
       
        return control_flow_ops.group(*[var_update, m_t, v_t, vhat])


   # @tf.function(jit_compile=True)
    def _apply_sparse_shared(self, grad, var, indices, scatter_add):
        beta1_power, beta2_power = self._get_beta_accumulators()
        beta1_power = math_ops.cast(beta1_power, var.dtype.base_dtype)
        beta2_power = math_ops.cast(beta2_power, var.dtype.base_dtype)
        lr_t_min = math_ops.cast(self._lr_t_min, var.dtype.base_dtype)
        lr_t_max = math_ops.cast(self._lr_t_max, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)
        adaptive = math_ops.cast(self._adaptive, var.dtype.base_dtype)
        lr_min = (lr_t_min * math_ops.sqrt(1 - beta2_power) / (1 - beta1_power))
        lr_max = (lr_t_max * math_ops.sqrt(1 - beta2_power) / (1 - beta1_power))
        # m_t = beta1 * m + (1 - beta1) * g_t
        m = self.get_slot(var, "m")
        m_scaled_g_values = grad * (1 - beta1_t)
        m_t = state_ops.assign(m, m * beta1_t,
                               use_locking=self._use_locking)
        with ops.control_dependencies([m_t]):
            m_t = scatter_add(m, indices, m_scaled_g_values)
        # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
        v = self.get_slot(var, "v")
        v_scaled_g_values = (grad * grad) * (1 - beta2_t)
        v_t = state_ops.assign(v, v * beta2_t, use_locking=self._use_locking)
        with ops.control_dependencies([v_t]):
            v_t = scatter_add(v, indices, v_scaled_g_values)
        vhat = self.get_slot(var, "vhat")
        vhat_t = state_ops.assign(vhat, math_ops.maximum(v_t, vhat))
        v_p = math_ops.sqrt(vhat_t)
        g_min_mean = ((tf.math.sign((tf.compat.v1.math.reduce_mean((vhat_t))) - (vhat_t)) + 1) /2)  
        g_max_mean =  ((tf.math.sign((vhat_t) - (tf.compat.v1.math.reduce_mean((vhat_t)))) + 1)/ 2)
          
            
       # float ( tf.math.count_nonzero(g_max)) <(float(tf.math.count_nonzero(g_min_mean))) and
        #@tf.function
        var_update=tf.cond(tf.math.less(( ( tf.math.count_nonzero(g_max_mean))), tf.cast(adaptive, int64)*  ((tf.math.count_nonzero(g_min_mean)))), lambda:  state_ops.assign_sub(var,
                                          lr_min * m_t / (v_p + epsilon_t), use_locking=self._use_locking), lambda:    state_ops.assign_sub(var, lr_max * m_t / (v_p + epsilon_t),
                                          use_locking=self._use_locking))
        
        
        return control_flow_ops.group(*[var_update, m_t, v_t, vhat])

    def _apply_sparse(self, grad, var):
        return self._apply_sparse_shared(
        grad.values, var, grad.indices,
        lambda x, i, v: state_ops.scatter_add(  # pylint: disable=g-long-lambda
            x, i, v, use_locking=self._use_locking))

    def _resource_scatter_add(self, x, i, v):
        with ops.control_dependencies(
        [resource_variable_ops.resource_scatter_add(
            x.handle, i, v)]):
            return x.value()

    def _resource_apply_sparse(self, grad, var, indices):
        return self._apply_sparse_shared(
        grad, var, indices, self._resource_scatter_add)

    def _finish(self, update_ops, name_scope):
    # Update the power accumulators.
        with ops.control_dependencies(update_ops):
            beta1_power, beta2_power = self._get_beta_accumulators()
        with ops.colocate_with(beta1_power):
            update_beta1 = beta1_power.assign(
                beta1_power * self._beta1_t, use_locking=self._use_locking)
            update_beta2 = beta2_power.assign(
                beta2_power * self._beta2_t, use_locking=self._use_locking)
            return control_flow_ops.group(*update_ops + [update_beta1, update_beta2],
                                  name=name_scope)

# tf.constant(1)


