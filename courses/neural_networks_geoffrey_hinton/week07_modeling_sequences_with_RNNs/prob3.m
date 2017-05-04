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


function [y_outputs, h_outputs] = doProb3(x_inputs, w_xh, w_hh, w_hy, h_bias, y_bias)
  % Given a [1xN] column matrix of inputs, produce the
  % Recurrent Neural Network computation

  num_time_steps = size(x_inputs, 2)
  h_inputs = zeros(1, num_time_steps)
  h_logits = zeros(1, num_time_steps)
  h_outputs = zeros(1, num_time_steps)
  y_outputs = zeros(1, num_time_steps)

  for T = 1:num_time_steps

    printf('\n\nT = %d:\n', T);
    x_input = x_inputs(:, T)
    h_input = h_inputs(:, T)
    printf('x_input: %d\n', x_input);
    printf('h_input: %d\n', h_input);

    [unit_logit] = get_logit(x_input, w_xh, h_input, w_hh, h_bias)
    h_outputs(1, T) = unit_logit
    printf('h_logit: %d\n', unit_logit);

    [logistic_unit_output] = get_logit_activation(unit_logit)
    h_logits(1, T) = logistic_unit_output
    printf('h_output: %d\n', logistic_unit_output);

    y_output = logistic_unit_output * w_hy + y_bias
    y_outputs(1, T) = y_output
    printf('y_output: %d\n', y_output);

  endfor

endfunction


[y_outputs, h_outputs] = doProb3(x_inputs, w_xh, w_hh, w_hy, h_bias, y_bias)

y_at_t_equals_1 = y_outputs(:, 2)
% gives -0.51174, incorrect

