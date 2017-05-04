# Lecture 7 Quiz - Problem 3
#

w_xh = 0.5
w_hh = -1.0
w_hy = -0.7
h_bias = -1.0
y_bias = 0.0
x_inputs = [9, 4, -2]


function [logistic_unit_output] = get_logit_activation(k)

logistic_unit_output = 1 / ( 1 + exp( -1 * k ) )

endfunction


function [unit_logit] = get_logit(x_input, w_xh, h_input, w_hh, h_bias)

unit_logit = x_input * w_xh + h_input * w_hh + h_bias

endfunction


function [y_outputs, h_outputs] = prob3(x_inputs)
% Given a [1xN] column matrix of inputs, produce the
% Recurrent Neural Network computation

num_inputs = size(x_inputs, 2)
h_inputs = zeros(1, num_time_steps)
h_logits = zeros(1, num_time_steps)
h_outputs = zeros(1, num_time_steps)

for T = 0:num_time_steps
  fprintf(1, 'T = %d\n', time_step);
  x_input = x_inputs[:, T]
  h_input = h_inputs[:, T]
  [unit_logit] = get_logit(x_input, w_xh, h_input, w_hh, h_bias)
  h_outputs[1, T] = unit_logit
  [logistic_unit_output] = get_logit_activation(unit_logit)
  h_logits[1, T] = logistic_unit_output
  WORK IN PROGRESS
endfor

endfunction


[y_outputs, h_outputs] = prob3(x_inputs)
