

def calc_neuron(t_inputs, t_weights, t_bias):
    """
    Calculates the output of a neuron.

    First it checks if the number of inputs
    and weights is equal:
    if its not equal then the function raises a ValueError.

    After the check the function loops for each element
    in the <t_inputs> list and multiplies it by its weight (in <t_weights>).

    When the loop stops, the bias is summed to the output.

    The output finally is returned by the function.

    Example:
    >>> calc_neuron([1, 2, 3], [0.2, 0.5, 1.7], 3)  # (1*0.2 + 2*0.5 + 3*1.7 + 3)
    9.3

    :param list t_inputs: list of neuron inputs
    :param list t_weights: list of connection weights to the neuron
    :param int t_bias: integer, bias of the neuron
    :return t_output: integer, output of the neuron
    :raises ValueError: <t_inputs> and <t_weights> must have the same len
    :rtype: int
    """

    # neuron output
    t_output = 0

    # number of inputs must match the number of weights
    if len(t_inputs) != len(t_weights):
        raise ValueError("<t_inputs> and <t_weights> must have the same len")

    # loop for each input and multiply it by its weight
    for i in range(len(t_inputs)):
        t_output += t_inputs[i] * t_weights[i]

    # sum the bias to the output
    t_output += t_bias

    return t_output


def calc_layer_by_neurons(t_inputs, t_neurons):
    """
    Calculates the output of a layer by passing the input data and neurons properties.

    The function loops for each element in the <t_neurons> list:
    each element is a dictionary and each element has to have <weights> and <bias> properties.
    > otherwise a ValueError error is raised

    The element's properties are then passed to the calc_neuron() function to calculate
    the output of each single neuron.

    The result of each element/neuron is then collected in the <t_output> list.

    Finally the <t_output> list is returned by the function

    :param list t_inputs: list of inputs
    :param list t_neurons: list of dictionaries, each with <weights> and <bias> properties
    :raises ValueError: raised when a neuron has no <weights> or no <bias>
    :return list t_output: list with the neuron outputs
    """

    # contains neurons outputs
    t_output = []

    # loop for each neuron in the layer
    for neuron in t_neurons:

        # be sure that <weights> and <bias> are in the neuron dictionary
        if "weights" not in neuron:
            raise ValueError("neuron '{}' must have <weights>".format(neuron))

        if "bias" not in neuron:
            raise ValueError("neuron '{}' must have <bias>".format(neuron))

        # save the data in variables
        neuron_weights = neuron["weights"]
        neuron_bias = neuron["bias"]

        # call the calc_neuron() function to calculate the output of the single neuron
        neuron_output = calc_neuron(t_inputs, neuron_weights, neuron_bias)

        # append the output to the layer output list
        t_output.append(neuron_output)

    return t_output


def calc_layer_by_props(t_inputs, t_m_weights, t_biases):
    """
    Calculates the output of a layer by passing the input data, a matrix with weights and biases.

    First it checks if the array of weights has one set per each bias (so one for each neuron...)
    and then a for loop creates the dictionaries to pass to the calc_layer_by_neurons() function.

    After the function call the output of the layer is returned.

    :param list t_inputs: list of inputs
    :param list t_m_weights: list of lists, each row is a set of connection weights to a neuron
    :param list t_biases: list of biases
    :return t_output: list of neurons outputs in one layer
    :rtype: list
    """

    if len(t_m_weights) != len(t_biases):
        # t_m_weights is a matrix with Y lists (where Y is the number of neurons)
        # of X elements (X is the number of connections to one neuron).
        # Y must match the number of biases because both are equal to the number of neurons
        raise ValueError("len of <t_m_weights> and <t_biases> must match")

    # memorize properties of neurons
    neurons_props = []

    # for each weight and bias properties there is a neuron
    for i in range(len(t_m_weights)):
        # save the weights of the connections to the neuron
        neuron_weights = t_m_weights[i]
        # save the bias of the neuron
        neuron_bias = t_biases[i]

        # save the properties of the single neuron in a dictionary and add it to the neurons properties
        neuron_props = {"weights": neuron_weights, "bias": neuron_bias}
        neurons_props.append(neuron_props)

    # calculate the output of the layer by passing data and neurons properties to
    # the calc_layer_by_neurons() function
    t_output = calc_layer_by_neurons(t_inputs, neurons_props)

    return t_output


def calc_batches_layer_by_props(t_m_inputs, t_m_weights, t_biases):
    """
    Calculates the output of a layer by passing batches of input data, a matrix with weights and biases.

    The function loops for each sample in the batch of inputs and calls the calc_layer_by_props() function
    to get the outputs of a layer.

    Each output of the layers is saved and then returned by the function.

    :param list t_m_inputs: list of lists, batches of input data
    :param list t_m_weights: list of lists, each row is a set of connection weights to a neuron
    :param list t_biases: list of biases
    :return layer_outputs: list of lists, contains the outputs of the layer of each sample data.
    :rtype: list
    """
    # memorize layer outputs
    layer_outputs = []

    # for each input sample, calculate the layer output
    for inputs in t_m_inputs:
        # calculate the layer output and add it to the layer outputs matrix
        layer_output = calc_layer_by_props(inputs, t_m_weights, t_biases)
        layer_outputs.append(layer_output)

    return layer_outputs


def neuron_relu_function(t_input):
    """
    Rectified linear activation function for one neuron.

    If the input (output of the neuron calculations) is less than 0, returns 0.
    Else returns t_input

    :param int t_input: input value
    :return output: output value (0 or t_input)
    :rtype: int
    """

    # default: output is equal to t_input
    output = t_input

    # if the input value is less than zero, set the output to zero
    if t_input < 0:
        output = 0

    return output


def layer_relu_function(t_inputs):
    """
    Rectified linear activation function for a layer.

    Returns a list of integers:
    a loop checks if each input is less than 0 or not
    thanks to the neuron_relu_function()
    and appends the returned value to the output list

    :param list t_inputs: list of inputs (neurons' outputs)
    :return output: list of ReLU outputs
    :rtype: list
    """

    # contains the outputs of the relu function
    output = []

    # loop for each neuron output of the layer
    for relu_input in t_inputs:
        # pass to the rectified linear function each output
        relu_output = neuron_relu_function(relu_input)

        # add each relu output to the output list
        output.append(relu_output)

    return output


def batch_layer_relu_function(t_m_inputs):
    """
    Rectified linear activation function for a layer, data passed in samples.

    A loop retrieves the output of a layer for each sample data
    and uses the ReLU function to set all the negative values to zero.

    :param list t_m_inputs: list of lists, outputs (many samples) of a layer
    :return output: list of lists, outputs of the ReLU function (all the samples, one layer)
    :rtype: list
    """

    # contains the relu output of all the samples
    output = []

    # for each sample output of the layer (result of every neuron in the layer, for all the samples)
    for sample_output in t_m_inputs:
        # calculate the ReLU output of the single sample output
        sample_relu_output = layer_relu_function(sample_output)

        # add the output to the relu outputs
        output.append(sample_relu_output)

    return output
