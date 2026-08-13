#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 10:29:58 2022

@author: david
"""

import jax
import jax.numpy as jnp

x = jnp.arange(10)

def sum_of_squares(x):
  return jnp.sum(x**2)

sum_of_squares_dx = jax.grad(sum_of_squares)

x = jnp.asarray([1.0, 2.0, 3.0, 4.0])

print(sum_of_squares(x))

print(sum_of_squares_dx(x))